import numpy as np
import torch
from torch.nn import Parameter

from stork import activations
from stork.nodes.base import CellGroup
from stork.utils import lif_membrane_dynamics

EPSILON = 1e-9


class FilterLIFGroup(CellGroup):
    def __init__(
        self,
        shape,
        nb_groups,
        tau_mem=10e-3,
        tau_filter=5e-3,
        nb_filters=5,
        diff_reset=False,
        learn_timescales=False,
        learn_A=False,
        nb_off_diag=1,
        clamp_mem=False,
        activation=activations.SuperSpike,
        dropout_p=0.0,
        stateful=False,
        name="FilterLIFGroup",
        regularizers=None,
        **kwargs
    ):
        """
        Leaky Integrate-and-Fire neuron with decaying synaptic input current.
        It has three state variables that are scalars and are updated at every time step:
        `mem` is for the membrane potential, `syn` is for the synaptic input current, and `out` is 0/1 depending on
        whether the neuron produces a spike.

        Args:
            :param shape: The number of units in this group
            :type shape: int or tuple of int
            :param tau_mem: The membrane time constant in s, defaults to 10e-3
            :type tau_mem: float
            :param tau_syn: The synaptic time constant in s, defaults to 5e-3
            :type tau_syn: float
            :param diff_reset: Whether or not to differentiate through the reset term, defaults to False
            :type diff_reset: bool
            :param learn_timescales: Whether to learn the membrane and synaptic time constants, defaults to False
            :type learn_timescales: bool
            :param activation: The surrogate derivative enabled activation function, defaults to stork.activations.SuperSpike
            :type activation: stork.activations
            :param dropout_p: probability that some elements of the input will be zeroed, defaults to 0.0
            :type dropout_p: float
            :param stateful: Whether or not to reset the state of the neurons between mini-batches, defaults to False
            :type stateful: bool
            :param regularizers: List of regularizers
        """

        super().__init__(
            shape,
            dropout_p=dropout_p,
            stateful=stateful,
            name=name,
            regularizers=regularizers,
            **kwargs
        )
        self.nb_groups = nb_groups
        self.tau_mem = tau_mem
        self.tau_filter = tau_filter
        self.nb_filters = nb_filters

        self.learn_A = learn_A
        self.nb_off_diag = nb_off_diag

        self.spk_nl = activation.apply
        self.diff_reset = diff_reset
        self.learn_timescales = learn_timescales
        self.clamp_mem = clamp_mem
        self.mem = None
        self.filt = None
        self.W_filt = None
        self.filt_param = None
        self.A = None

        # Group membership tensor
        assert (
            self.nb_units % self.nb_groups == 0
        ), "Nb_units must be divisible by nb_groups"
        self.nb_units_per_group = self.nb_units // self.nb_groups
        self.group_membership = torch.tensor(
            list(range(self.nb_groups)) * self.nb_units_per_group
        )

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        self.dcy_mem = float(np.exp(-time_step / self.tau_mem))
        self.scl_mem = 1.0 - self.dcy_mem

        # create A matrix
        self.dcy_filter = float(np.exp(-time_step / self.tau_filter))
        self.scl_filter = 1 - self.dcy_filter

        A_shape = (self.nb_filters, self.nb_filters)
        self.A = torch.zeros(A_shape, device=device, dtype=dtype)
        self.A.fill_diagonal_(self.dcy_filter)

        for d in range(1, self.nb_off_diag + 1):
            for i in range(self.nb_filters - d):
                self.A[i, i + d] = self.scl_filter

        filt_param = torch.randn(
            (self.nb_groups, self.nb_filters),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        self.filt_param = Parameter(filt_param, requires_grad=True)

        if self.learn_timescales:
            mem_param = torch.randn(1, device=device, dtype=dtype, requires_grad=True)
            self.mem_param = Parameter(mem_param, requires_grad=self.learn_timescales)

        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        if self.learn_timescales:
            self.dcy_mem = torch.exp(
                -self.time_step / (2 * self.tau_mem * torch.sigmoid(self.mem_param))
            )
            self.scl_mem = 1.0 - self.dcy_mem

        self.filter_shape = (batch_size, *self.shape, self.nb_filters)

        self.filt = torch.zeros(
            *self.filter_shape, device=self.device, dtype=self.dtype
        )

        # self.W_filt = self.filt_param.repeat(self.nb_units_per_group, 1).to(self.device)

        # self.W_filt = self.filt_param.unsqueeze(0).expand(
        #     self.nb_units_per_group, self.nb_groups, self.nb_filters
        # ).flatten(start_dim=0, end_dim=1)

        self.W_filt = (
            self.filt_param.expand(
                size=(self.nb_units_per_group, *self.filt_param.shape)
            )
            .reshape(shape=(self.nb_units, self.nb_filters))
            .to(self.device)
        )

        self.mem = self.get_state_tensor("mem", state=self.mem)
        self.out = self.states["out"] = torch.zeros(
            self.int_shape, device=self.device, dtype=self.dtype
        )

    def get_spike_and_reset(self, mem):
        mthr = mem - 1.0
        out = self.spk_nl(mthr)

        if self.diff_reset:
            rst = out
        else:
            # if differentiation should not go through reset term, detach it from the computational graph
            rst = out.detach()

        return out, rst

    def forward(self):
        # spike & reset
        new_out, rst = self.get_spike_and_reset(self.mem)

        # synaptic dynamics
        # Update filters (z=f)
        new_filt = self.filt @ self.A

        # add spiketrain to first filters
        new_filt[:, :, 0].add_(self.input)

        syn_input = torch.einsum("nf,bnf->bn", self.W_filt, new_filt)

        # membrane dynamics
        new_mem = (self.dcy_mem * self.mem + self.scl_mem * syn_input) * (
            1.0 - rst
        )  # multiplicative reset

        # Clamp membrane potential
        if self.clamp_mem:
            new_mem = torch.clamp(new_mem, max=1.01)

        self.out = self.states["out"] = new_out
        self.mem = self.states["mem"] = new_mem
        self.filt = new_filt

    def get_filterbanks_by_group(
        self, sum_over_filters=True, T=None, A=None, filt_param=None
    ):
        """
        Used to inspect the learned synaptic filters.
        Returns an object of shape (time, nb_groups)
        that contains the weighted filterbanks for each group.

        If sum_over_filters=False, returns an object of shape (time, nb_groups, nb_filters)
        with each individually weighted filter.
        """

        if filt_param is None:
            filt_param = self.filt_param

        if A is None:
            A = self.A

        if T is None:
            T = self.nb_filters * self.tau_filter * 5

        # Prepare empty tensors
        timesteps = int(T / self.time_step)

        # Filter state
        filter_shape = (self.nb_groups, self.nb_filters)
        filt_state = torch.zeros(filter_shape, device=self.device, dtype=self.dtype)

        # Output
        output = torch.zeros(
            (timesteps + 1, *filter_shape), device=self.device, dtype=self.dtype
        )

        # Evolve filters over time
        filt_state[..., 0] = 1
        output[0] = filt_state

        for timestep in range(timesteps):

            # Update filters (z = f)
            filt_state = torch.einsum("gf,fz->gz", filt_state, A)
            output[timestep + 1] = filt_state

        # Weight the filterbanks
        w = filt_param.clone().detach()  # shape [nb_groups, nb_filters]
        output_weighted = w.view(1, *w.shape) * output

        if sum_over_filters:
            return output_weighted.sum(-1).cpu().detach().numpy()

        else:
            return output_weighted.cpu().detach().numpy()

    def get_membrane_kernel_by_group(self, time_step=None):
        """
        Used to inspect the learned PSP kernels.
        Returns an object of shape (time, nb_groups)
        that contains the post-synaptic potential kernel for each group.

        I this is called before `model.configure` is called, it will draw random filter
        weights for each group.
        """

        device = torch.device("cpu")
        dtype = torch.float

        if time_step == None:
            time_step = self.time_step

        if self.filt_param == None:
            filt_param = torch.randn(
                (self.nb_groups, self.nb_filters), device=device, dtype=dtype
            )
        else:
            filt_param = self.filt_param.clone().detach().cpu()

        if self.A == None:

            dcy_filter = float(np.exp(-time_step / self.tau_filter))
            scl_filter = 1 - self.dcy_filter

            A_shape = (self.nb_filters, self.nb_filters)
            A = torch.zeros(A_shape, device=device, dtype=dtype)
            A.fill_diagonal_(dcy_filter)

            for d in range(1, self.nb_off_diag + 1):
                for i in range(self.nb_filters - d):
                    self.A[i, i + d] = scl_filter

        else:
            A = self.A.clone().detach().cpu()

        # Compute filterbanks
        T = self.nb_filters * self.tau_filter * 5

        syn_currents = self.get_filterbanks_by_group(
            sum_over_filters=True, T=T, A=A, filt_param=filt_param
        )

        mem_by_group = lif_membrane_dynamics(syn_currents, self.tau_mem, dt=time_step)

        return mem_by_group

    def get_epsilon_numerical(self, time_step):
        """
        Returns numerical values of epsilon_bar and epsilon_hat
        (used by fluctuation-driven initialization strategies)
        """

        # shape [time, nb_groups]
        kernels = self.get_membrane_kernel_by_group(time_step)

        epsilon_bar = (kernels.sum(0) * time_step).mean() + EPSILON
        epsilon_hat = ((kernels**2).sum(0) * time_step).mean()

        return epsilon_bar, epsilon_hat
