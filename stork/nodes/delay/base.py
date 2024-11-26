import numpy as np
import torch
from torch.nn import Parameter

from stork import activations
from stork.nodes.base import CellGroup


class SGQuantization(torch.autograd.Function):
    """
    Autograd for Surrogate gradients of quantized delays
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.round(input.clip(0)).long()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = torch.heaviside(input, 0.0)
        return grad_output * grad_input


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

        # precompute indices
        self.neuron_indices = torch.arange(src.shape[-1])

        # init parameters (delay per neuron)
        self.reset_parameters()

    def reset_parameters(self):
        # initializes learnable delays
        self.delays = Parameter(torch.rand(self.shape)*self.max_delay_timesteps, requires_grad=True)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)

        self.out = self.states["out"] = torch.zeros(self.int_shape, device=self.device, dtype=self.dtype)

        self.buffer_shape = (self.max_delay_timesteps,) + self.int_shape
        self.buffer = self.states['buffer'] =torch.zeros(self.buffer_shape, device=self.device, dtype=self.dtype)

        # convert float delays to integer indices
        self.delay_indices = SGQuantization.apply(self.delays)


    def forward(self):

        # update bufer
        self.buffer[self.clk % self.max_delay_timesteps] = self.src.out

        # compute valid neurons
        valid_neurons = self.delay_indices <= self.clk

        # Init output tensor with zeros
        self.out = self.states["out"] = torch.zeros(self.int_shape, device=self.device, dtype=self.dtype)

        if valid_neurons.any():
           # Compute buffer indices for all neurons
            buffer_indices = (self.clk - self.delay_indices) % self.max_delay_timesteps  # Shape: [N_neurons]

            # Gather the delayed data for all neurons
            # Indexing over the first and third dimensions
            delayed_data = self.buffer[buffer_indices, :, self.neuron_indices]  # Shape: [N_neurons, N_batch]

            # Transpose to get shape [N_batch, N_neurons]
            delayed_data = delayed_data.transpose(0, 1)  # Shape: [N_batch, N_neurons]

            # Zero out the data for invalid neurons
            delayed_data[:, ~valid_neurons] = 0  # Invalid neurons set to zero

            # Assign to output
            self.out = self.states["out"] = delayed_data

