import numpy as np
import torch
from torch.nn import Parameter

from stork import activations
from stork.nodes.base import CellGroup


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
        name="LIFGroup",
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

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        self.dcy_mem = float(np.exp(-time_step / self.tau_mem))
        self.scl_mem = 1.0 - self.dcy_mem

        # create A matrix
        self.dcy_filter = float(np.exp(-time_step / self.tau_filter))
        self.scl_filter = 1 - self.dcy_filter

        A_shape = (self.nb_filters, self.nb_filters)
        self.A = torch.zeros(A_shape, device=device, dtype=dtype)
        self.A.fill_diagonal_(self.dcy_filter)

        for d in range(self.nb_off_diag):
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

        self.filt = torch.zeros(*self.filter_shape, device=self.device, dtype=self.dtype)

        self.W_filt = self.filt_param.expand(
            size=(self.nb_units // self.nb_groups, *self.filt_param.shape)
        ).reshape(shape=(self.nb_units, self.nb_filters))

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
        # Update filters
        new_filt = torch.einsum("bnf,fg->bng", self.filt, self.A)
        # add spiketrain to first filters
        new_filt[:, :, 0] += self.input
        # TODO: check whether the input is added correctly

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
