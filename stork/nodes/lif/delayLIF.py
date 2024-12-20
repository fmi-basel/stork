import numpy as np
import torch
from torch.nn import Parameter

from stork import activations
from stork.nodes.base import CellGroup


class SGQuantization(torch.autograd.Function):
    """
    Autograd for surrogate gradients of quantized delays
    """

    @staticmethod
    def forward(
        ctx,
        delays,
        mem_buffer,
        spk_buffer,
        delayed_mem,
        delayed_spk,
        clk,
        max_delay_timesteps,
        neuron_indices,
    ):
        """In the forward pass, we quantize the received delays and read the delayed output from the buffer,
        which we then forward as the new output.

        Args:
            ctx (_type_): The self of the forward pass
            mem_buffer (torch.floattensor): Cyclic buffer, that saves the current mem
            mem_buffer (torch.floattensor): Cyclic buffer, that saves the current spikes
            delays (torch.floattensor): Not rounded delays in time steps (float values allowed)
            delayed_mem (torch.floattensor): Placeholder to be filled with delayed mem
            delayed_spk (torch.floattensor): Placeholder to be filled with delayed output spikes
            clk (int): The current time step
            max_delay_timesteps (int): The maximal delay (in time steps)
            neuron_indices (torch.tensor): A tensor with the index of each neuron

        Returns:
            _type_: _description_
        """

        # convert float delays to integer indices
        delay_indices = torch.clamp(torch.round(delays), 0).long()

        # compute valid neurons
        valid_neurons = delay_indices <= clk

        if valid_neurons.any():
            # Compute buffer indices for all neurons
            buffer_indices = (
                clk - delay_indices
            ) % max_delay_timesteps  # Shape: [N_neurons]

            # Gather the delayed data for all neurons
            # Indexing over the first and third dimensions
            delayed_spk = spk_buffer[
                buffer_indices, :, neuron_indices
            ]  # Shape: [N_neurons, N_batch]
            delayed_mem = mem_buffer[buffer_indices, :, neuron_indices]

            # Transpose to get shape [N_batch, N_neurons]
            delayed_spk = delayed_spk.transpose(0, 1)  # Shape: [N_batch, N_neurons]
            delayed_mem = delayed_mem.transpose(0, 1)

            # Zero out the data for invalid neurons
            delayed_spk[:, ~valid_neurons] = 0  # Invalid neurons set to zero
            delayed_mem[:, ~valid_neurons] = 0

        # print("forward")
        ctx.save_for_backward(delays, delayed_mem)

        # check which torch dtype is returned

        return delayed_spk

    @staticmethod
    def backward(ctx, grad_output):
        """Here we provide a backward for the quantized delay. We assume, the forward is a coarse relu function,
        hence the backward would be the derivative of a relu."""
        (delays, delayed_mem) = ctx.saved_tensors

        grad_input = (delays > 0) * delayed_mem

        return grad_output * grad_input, None, None, None, None, None, None, None


class DelayLIF(CellGroup):
    def __init__(
        self,
        shape,
        tau_mem=10e-3,
        tau_syn=5e-3,
        max_delay_timesteps=100,
        diff_reset=False,
        learn_tau_mem=False,
        learn_tau_syn=False,
        learn_tau_mem_hetero=False,
        learn_tau_syn_hetero=False,
        clamp_mem=False,
        activation=activations.SuperSpike,
        dropout_p=0.0,
        stateful=False,
        name="DelayLIFGroup",
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

        self.tau_mem = torch.tensor(tau_mem)
        self.tau_syn = torch.tensor(tau_syn)
        self.spk_nl = activation.apply
        self.diff_reset = diff_reset
        self.learn_tau_mem = learn_tau_mem
        self.learn_tau_syn = learn_tau_syn
        self.learn_tau_mem_hetero = learn_tau_mem_hetero
        self.learn_tau_syn_hetero = learn_tau_syn_hetero
        self.clamp_mem = clamp_mem
        self.mem = None
        self.syn = None

        # Assert that tau_mem and tau_syn are either both scalars or both arrays of shape `shape`
        assert self.tau_mem.shape == torch.Size([]) or self.tau_mem.shape == torch.Size(
            [shape]
        )
        assert self.tau_syn.shape == torch.Size([]) or self.tau_syn.shape == torch.Size(
            [shape]
        )

        # Initialize mem_param and syn_param if learning
        if self.learn_tau_mem:
            if self.learn_tau_mem_hetero:
                self.mem_param_shape = self.shape
            else:
                self.mem_param_shape = 1

        if self.learn_tau_syn:
            if self.learn_tau_syn_hetero:
                self.syn_param_shape = self.shape
            else:
                self.syn_param_shape = 1

        self.max_delay_timesteps = max_delay_timesteps
        self.delay_nl = SGQuantization.apply

        # precompute indices
        self.neuron_indices = torch.arange(shape)

        self.reset_parameters()

    def reset_parameters(self):
        if self.learn_tau_mem:
            mem_param = torch.randn(self.mem_param_shape)
            mem_param = mem_param / 4 + 1
            self.mem_param = Parameter(mem_param, requires_grad=True)

        if self.learn_tau_syn:
            syn_param = torch.randn(self.syn_param_shape)
            syn_param = syn_param / 4 + 1
            self.syn_param = Parameter(syn_param, requires_grad=True)

        # initializes learnable delays
        self.delays = Parameter(
            torch.rand(self.shape) * self.max_delay_timesteps,
            requires_grad=True,
        )

    def configure(self, batch_size, nb_steps, time_step, device, dtype):

        # Set device and dtype for tau_mem and tau_syn
        self.tau_mem = self.tau_mem.to(device=device, dtype=dtype)
        self.tau_syn = self.tau_syn.to(device=device, dtype=dtype)

        self.delays = self.delays.to(device=device, dtype=dtype)
        self.to(device=device, dtype=dtype)  # This moves parameters

        self.dcy_mem = torch.exp(-time_step / self.tau_mem)
        self.scl_mem = 1.0 - self.dcy_mem
        self.dcy_syn = torch.exp(-time_step / self.tau_syn)
        self.scl_syn = 1.0 - self.dcy_syn

        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)

        if self.learn_tau_mem:
            time_step = torch.tensor(
                self.time_step, device=self.device, dtype=self.dtype
            )
            self.dcy_mem = torch.exp(
                -time_step
                / (self.tau_mem * torch.nn.functional.softplus(self.mem_param))
            )
            self.scl_mem = 1.0 - self.dcy_mem

        if self.learn_tau_syn:
            if "time_step" not in locals():
                time_step = torch.tensor(
                    self.time_step, device=self.device, dtype=self.dtype
                )
            self.dcy_syn = torch.exp(
                -time_step
                / (self.tau_syn * torch.nn.functional.softplus(self.syn_param))
            )
            self.scl_syn = 1.0 - self.dcy_syn

        # Delay buffer
        self.buffer_shape = (self.max_delay_timesteps,) + self.int_shape
        self.mem_buffer = self.states["mem_buffer"] = torch.zeros(
            self.buffer_shape, device=self.device, dtype=self.dtype
        )
        self.spk_buffer = self.states["spk_buffer"] = torch.zeros(
            self.buffer_shape, device=self.device, dtype=self.dtype
        )

        self.mem = self.get_state_tensor("mem", state=self.mem)
        self.syn = self.get_state_tensor("syn", state=self.syn)
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

        # synaptic & membrane dynamics
        new_syn = self.dcy_syn * self.syn + self.input
        new_mem = (self.dcy_mem * self.mem + self.scl_mem * self.syn) * (
            1.0 - rst
        )  # multiplicative reset

        # Clamp membrane potential
        if self.clamp_mem:
            new_mem = torch.clamp(new_mem, max=1.01)

        self.mem = self.states["mem"] = new_mem
        self.syn = self.states["syn"] = new_syn

        # update mem buffer for current time step from the output of the source group
        self.mem_buffer[self.clk % self.max_delay_timesteps] = new_mem
        # update buffer for current time step from the output of the source group
        self.spk_buffer[self.clk % self.max_delay_timesteps] = new_out

        # Init output tensor with zeros
        delayed_mem = torch.zeros(self.int_shape, device=self.device, dtype=self.dtype)
        delayed_spk = torch.zeros(self.int_shape, device=self.device, dtype=self.dtype)

        # Compute delayed output
        self.out = self.states["out"] = self.delay_nl(
            self.delays,
            self.mem_buffer,
            self.spk_buffer,
            delayed_mem,
            delayed_spk,
            self.clk,
            self.max_delay_timesteps,
            self.neuron_indices,
        )
