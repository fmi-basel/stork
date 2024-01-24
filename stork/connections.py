import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
import torch.nn.functional as F

import numpy as np

from . import core
from . import constraints as stork_constraints

from . import operations


class BaseConnection(core.NetworkNode):
    def __init__(
        self, src, dst, target=None, name=None, regularizers=None, constraints=None
    ):
        """Abstract base class of Connection objects.

        Args:
            src (CellGroup): The source group
            dst (CellGroup: The destination group
            target (string, optional): The name of the target state tensor.
            name (string, optional): Name of the node
            regularizers (list): List of regularizer objects.
            constraints (list): List of constraints.

        """

        super(BaseConnection, self).__init__(name=name, regularizers=regularizers)
        self.src = src
        self.dst = dst

        if target is None:
            self.target = dst.default_target
        else:
            self.target = target

        if constraints is None:
            self.constraints = []
        elif type(constraints) == list:
            self.constraints = constraints
        elif issubclass(type(constraints), stork_constraints.WeightConstraint):
            self.constraints = [constraints]
        else:
            raise ValueError

    def init_parameters(self, initializer):
        """
        Initializes connection weights and biases.
        """
        initializer.initialize(self)
        self.apply_constraints()

    def propagate(self):
        raise NotImplementedError

    def apply_constraints(self):
        raise NotImplementedError


class Connection(BaseConnection):
    def __init__(
        self,
        src,
        dst,
        operation=None,
        target=None,
        bias=False,
        requires_grad=True,
        propagate_gradients=True,
        flatten_input=False,
        name=None,
        regularizers=None,
        constraints=None,
        enable_delays=False,
        learn_delays=False,
        **kwargs
    ):
        super(Connection, self).__init__(
            src,
            dst,
            name=name,
            target=target,
            regularizers=regularizers,
            constraints=constraints,
        )

        self.requires_grad = requires_grad
        self.propagate_gradients = propagate_gradients
        self.flatten_input = flatten_input
        self.enable_delays = enable_delays
        self.learn_delays = learn_delays

        if flatten_input:
            src_shape = src.nb_units
        else:
            src_shape = src.shape[0]

        if self.enable_delays:
            self.delays = Parameter(
                torch.zeros(dst.nb_units, src_shape), requires_grad=learn_delays
            )
            if operation is None:
                operation = operations.MaskedLinear
        else:
            if operation is None:
                operation = nn.Linear
            self.delays = None

        self.op = operation(src_shape, dst.shape[0], bias=bias, **kwargs)
        for param in self.op.parameters():
            param.requires_grad = requires_grad

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)
        self.prebatch_hook()

    def add_diagonal_structure(self, width=1.0, ampl=1.0):
        if type(self.op) != nn.Linear:
            raise ValueError("Expected op to be nn.Linear to add diagonal structure.")
        A = np.zeros(self.op.weight.shape)
        x = np.linspace(0, A.shape[0], A.shape[1])
        for i in range(len(A)):
            A[i] = ampl * np.exp(-((x - i) ** 2) / width**2)
        self.op.weight.data += torch.from_numpy(A)

    def get_weights(self):
        return self.op.weight

    def get_delays(self):
        return self.delays

    def set_delays(self, delays):
        self.delays = Parameter(delays, requires_grad=self.learn_delays)

    def get_regularizer_loss(self):
        reg_loss = torch.tensor(0.0, device=self.device)
        for reg in self.regularizers:
            reg_loss += reg(self.get_weights())
        return reg_loss

    def prebatch_hook(self):
        if self.enable_delays:
            int_delays = (self.delays / self.time_step).long()
            self.binary_delay_kernel = F.one_hot(int_delays).unsqueeze(0)
            self.count = torch.zeros(
                size=(self.batch_size, *self.binary_delay_kernel.shape[1:]),
                device=self.device,
            )

    def apply_delays(self, preact):
        self.count += preact.unsqueeze(1).unsqueeze(-1) * self.binary_delay_kernel

        mask = self.count[..., 0].clone()
        self.count[..., 0] = 0
        self.count = torch.roll(self.count, -1, -1)

        return mask

    def forward(self):
        preact = self.src.out
        if self.enable_delays:
            mask = self.apply_delays(preact)
            preact = torch.ones_like(preact)

        if not self.propagate_gradients:
            preact = preact.detach()
        if self.flatten_input:
            shp = preact.shape
            preact = preact.reshape(shp[:1] + (-1,))

        if self.enable_delays:
            out = self.op(preact, mask)
        else:
            out = self.op(preact)
        self.dst.add_to_state(self.target, out)

    def propagate(self):
        self.forward()

    def apply_constraints(self):
        for const in self.constraints:
            const.apply(self.op.weight)


class IdentityConnection(BaseConnection):
    def __init__(
        self,
        src,
        dst,
        target=None,
        bias=False,
        requires_grad=True,
        name=None,
        regularizers=None,
        constraints=None,
        tie_weights=None,
        weight_scale=1.0,
    ):
        """Initialize IdentityConnection

        Args:
            tie_weights (list of int, optional): Tie weights along dims given in list
            weight_scale (float, optional): Scale everything by this factor. Useful when the connection is used for relaying currents rather than spikes.
        """
        super(IdentityConnection, self).__init__(
            src,
            dst,
            name=name,
            target=target,
            regularizers=regularizers,
            constraints=constraints,
        )

        self.requires_grad = requires_grad
        self.weight_scale = weight_scale
        wshp = src.shape

        # Set weights tensor dimension to 1 along tied dimensions
        if tie_weights is not None:
            wshp = list(wshp)
            for d in tie_weights:
                wshp[d] = 1
            wshp = tuple(wshp)

        self.weights = Parameter(torch.randn(wshp), requires_grad=requires_grad)
        if bias:
            self.bias = Parameter(torch.randn(wshp), requires_grad=requires_grad)

    def get_weights(self):
        return self.weights

    def get_regularizer_loss(self):
        reg_loss = torch.tensor(0.0, device=self.device)
        for reg in self.regularizers:
            reg_loss += reg(self.get_weights())
        return reg_loss

    def apply_constraints(self):
        for const in self.constraints:
            const.apply(self.weights)

    def forward(self):
        preact = self.src.out
        if self.bias is None:
            self.dst.scale_and_add_to_state(
                self.weight_scale, self.target, self.weights * preact
            )
        else:
            self.dst.scale_and_add_to_state(
                self.weight_scale, self.target, self.weights * preact + self.bias
            )

    def propagate(self):
        self.forward()


class ConvConnection(Connection):
    def __init__(self, src, dst, conv=nn.Conv1d, **kwargs):
        super(ConvConnection, self).__init__(src, dst, operation=conv, **kwargs)


class Conv2dConnection(Connection):
    def __init__(self, src, dst, conv=nn.Conv2d, **kwargs):
        super(Conv2dConnection, self).__init__(src, dst, operation=conv, **kwargs)
