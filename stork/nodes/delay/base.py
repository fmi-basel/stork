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
        ctx, buffer, delays, delayed_data, clk, max_delay_timesteps, neuron_indices
    ):
        """In the forward pass, we quantize the received delays and read the delayed output from the buffer,
        which we then forward as the new output.

        Args:
            ctx (_type_): The self of the forward pass
            buffer (torch.floattensor): Cyclic buffer, that saves the current output of the previous layer
            delays (torch.floattensor): Not rounded delays in time steps (float values allowed)
            delayed_data (torch.floattensor): Placeholder to be filled with delayed output
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
            delayed_data = buffer[
                buffer_indices, :, neuron_indices
            ]  # Shape: [N_neurons, N_batch]

            # Transpose to get shape [N_batch, N_neurons]
            delayed_data = delayed_data.transpose(0, 1)  # Shape: [N_batch, N_neurons]

            # Zero out the data for invalid neurons
            delayed_data[:, ~valid_neurons] = 0  # Invalid neurons set to zero

        # print("forward")
        ctx.save_for_backward(delays, delayed_data)

        return delayed_data

    @staticmethod
    def backward(ctx, grad_output):
        """Here we provide a backward for the quantized delay. We assume, the forward is a coarse relu function,
        hence the backward would be the derivative of a relu."""
        (delays, delayed_data) = ctx.saved_tensors
        grad_input = (delays > 0) * delayed_data

        return None, grad_output * grad_input * -1, None, None, None, None


class DelayGroup(CellGroup):
    """
    Essentially a wrapper of src.out of a neuron (e.g. LIF) group that implements
    a delay buffer. Delays are per-neuron and learnable (i.e. trainable) parameters.
    """

    def __init__(
        self,
        src,
        max_delay_timesteps,
        store_sequences=None,
        name="DelayGroup",
        regularizers=None,
        dropout_p=0.0,
        stateful=False,
    ):
        shape = src.shape
        super(DelayGroup, self).__init__(
            shape,
            store_sequences,
            name,
            regularizers,
            dropout_p,
            stateful,
        )

        self.src = src
        self.max_delay_timesteps = max_delay_timesteps

        self.delay_nl = SGQuantization.apply

        # precompute indices
        self.neuron_indices = torch.arange(src.shape[-1])

        # init parameters (delay per neuron)
        self.reset_parameters()

    def reset_parameters(self):
        # initializes learnable delays
        self.delays = Parameter(
            torch.rand(self.shape) * self.max_delay_timesteps, requires_grad=True
        )

        pass

    def configure(self, batch_size, nb_steps, time_step, device, dtype):

        self.to(device=device, dtype=dtype)

        return super().configure(batch_size, nb_steps, time_step, device, dtype)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)

        self.buffer_shape = (self.max_delay_timesteps,) + self.int_shape
        self.buffer = self.states["buffer"] = torch.zeros(
            self.buffer_shape, device=self.device, dtype=self.dtype
        )

        self.out = self.states["out"] = torch.zeros(
            self.int_shape, device=self.device, dtype=self.dtype, requires_grad=True
        )

    def forward(self):

        # update bufer for current time step from the output of the source group
        self.buffer[self.clk % self.max_delay_timesteps] = self.src.out
        # Init output tensor with zeros
        out = torch.zeros(self.int_shape, device=self.device, dtype=self.dtype)

        # Compute delayed output
        self.out = self.states["out"] = self.delay_nl(
            self.buffer,
            self.delays,
            out,
            self.clk,
            self.max_delay_timesteps,
            self.neuron_indices,
        )
