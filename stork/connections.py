import torch
import torch.nn as nn
from torch.nn.modules import Linear

from torch.nn.parameter import Parameter

import numpy as np

from . import core
from . import constraints as stork_constraints


class BaseConnection(core.NetworkNode):
    def __init__(self, src, dst, target=None, name=None, regularizers=None, constraints=None):
        """ Abstract base class of Connection objects. 

        Args:
            src (CellGroup): The source group
            dst (CellGroup: The destination group
            target (string, optional): The name of the target state tensor.
            name (string, optional): Name of the node
            regularizers (list): List of regularizer objects.
            constraints (list): List of constraints.

        """

        super(BaseConnection, self).__init__(
            name=name, regularizers=regularizers)
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
    
    def reset_state(self, batchsize):
        pass


class Connection(BaseConnection):
    def __init__(self, src, dst, operation=nn.Linear, target=None, bias=False, requires_grad=True,
                 propagate_gradients=True, flatten_input=False, name=None, regularizers=None, constraints=None, **kwargs):
        super(Connection, self).__init__(src, dst, name=name,
                                         target=target, regularizers=regularizers, constraints=constraints)

        self.requires_grad = requires_grad
        self.propagate_gradients = propagate_gradients
        self.flatten_input = flatten_input

        if flatten_input:
            self.op = operation(
                src.nb_units, dst.shape[0], bias=bias, **kwargs)
        else:
            self.op = operation(
                src.shape[0], dst.shape[0], bias=bias, **kwargs)
        for param in self.op.parameters():
            param.requires_grad = requires_grad

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def add_diagonal_structure(self, width=1.0, ampl=1.0):
        if type(self.op) != nn.Linear:
            raise ValueError(
                'Expected op to be nn.Linear to add diagonal structure.')
        A = np.zeros(self.op.weight.shape)
        x = np.linspace(0, A.shape[0], A.shape[1])
        for i in range(len(A)):
            A[i] = ampl * np.exp(-(x - i) ** 2 / width ** 2)
        self.op.weight.data += torch.from_numpy(A)

    def get_weights(self):
        return self.op.weight

    def get_regularizer_loss(self):
        reg_loss = torch.tensor(0.0, device=self.device)
        for reg in self.regularizers:
            reg_loss += reg(self.get_weights())
        return reg_loss

    def forward(self):
        preact = self.src.out
        if not self.propagate_gradients:
            preact = preact.detach()
        if self.flatten_input:
            shp = preact.shape
            preact = preact.reshape(shp[:1] + (-1,))

        out = self.op(preact)
        self.dst.add_to_state(self.target, out)

    def propagate(self):
        self.forward()

    def apply_constraints(self):
        for const in self.constraints:
            const.apply(self.op.weight)
            
            
class SuperConnection(BaseConnection):
    def __init__(self, src, dst, tau_filter=10e-3, nb_filters=5, operation=nn.Linear, target=None, bias=False, requires_grad=True,
                 propagate_gradients=True, flatten_input=False, name=None, regularizers=None, constraints=None, **kwargs):
        super(SuperConnection, self).__init__(src, dst, name=name,
                                         target=target, regularizers=regularizers, constraints=constraints)

        self.requires_grad = requires_grad
        self.propagate_gradients = propagate_gradients
        self.flatten_input = flatten_input
        
        self.tau_filter = tau_filter
        self.nb_filters = nb_filters
        
        if flatten_input:
            self.src_shape = src.nb_units
        else:
            self.src_shape = src.shape[0]
        
        self.op = operation(
            self.src_shape * self.nb_filters, dst.shape[0], bias=bias, **kwargs)
        
        for param in self.op.parameters():
            param.requires_grad = requires_grad
            
        # State for filters
        self.filters = None
        
        # Update matrix for filters
        self.filter_update = None
        
    def reset_state(self, batchsize):
        
        if self.flatten_input:
            self.filter_shape = (batchsize, self.src.nb_units, self.nb_filters)
        else:
            self.filter_shape = (batchsize, *self.src.shape, self.nb_filters)
            
        self.filters = torch.zeros(self.filter_shape, device=self.device, dtype=self.dtype)

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        self.dcy_filter = float(np.exp(-time_step / self.tau_filter))
        self.scl_filter = 1 - self.dcy_filter
        
        # Make update matrix
        upd_shape = (self.nb_filters, self.nb_filters)
        self.filter_update = torch.zeros(upd_shape, device=device, dtype=dtype)
        self.filter_update.fill_diagonal_(self.dcy_filter)
        for i in range(self.nb_filters - 1):
            self.filter_update[i, i+1] = self.scl_filter
        
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def add_diagonal_structure(self, width=1.0, ampl=1.0):
        # This needs to be implemented to take into account the
        # filter structure of the weight matrix
        
        # Do we even need this ever?
        
        raise NotImplementedError
    
    def get_weights(self, return_3d=False):
        if return_3d:
            raise NotImplementedError
        return self.op.weight

    def get_regularizer_loss(self):
        reg_loss = torch.tensor(0.0, device=self.device)
        for reg in self.regularizers:
            reg_loss += reg(self.get_weights())
        return reg_loss        

    def forward(self):
        
        preact = self.src.out
        
        # Update filters
        new_filters = torch.einsum('bnf,fg->bng', self.filters, self.filter_update)
        
        # add spiketrain to first filters
        new_filters[:,:,0] += preact
        
        # OLD: UPDATE USING FOR LOOP
        # # # # # # # # # # # # # # # 
        
        #update = preact
        #for filt_idx in range(self.nb_filters):
            
        #    new_filters[:,:,filt_idx] = self.dcy_filter * self.filters[:,:,filt_idx] + update
        #    update = self.scl_filter * new_filters[:,:,filt_idx]
        
        self.filters = new_filters
                    
        filter_out = new_filters.view(self.filters.shape[0], 
                                       self.src_shape * self.nb_filters)
        
        if not self.propagate_gradients:
            filter_out = filter_out.detach()
        
        if self.flatten_input:
            shp = filter_out.shape
            filter_out = filter_out.reshape(shp[:1] + (-1,))

        out = self.op(filter_out)
        self.dst.add_to_state(self.target, out)

    def propagate(self):
        self.forward()

    def apply_constraints(self):
        for const in self.constraints:
            const.apply(self.op.weight)
            
    def get_filterbanks(self, T=1):
        """
        Used to inspect the learned synaptic filters.
        Returns an object of shape (time, n_pre, n_post)
        that contains the weighted filterbanks for each connection.
        
        If sum_over_filters=False, returns an object of shape (time, n_pre, n_post, n_filters)
        with each individually weighted filter.
        """
        
        # Prepare empty tensors
        timesteps = int(T / self.time_step)
        
        filter_shape = self.filter_shape[1:]            
        filter_state = torch.zeros(filter_shape, device=self.device, dtype=self.dtype)
        output = torch.zeros((timesteps + 1, *filter_shape), device=self.device, dtype=self.dtype)

        # Evolve filters over time
        filter_state[..., 0] = 1
        output[0] = filter_state
        
        for timestep in range(timesteps):
            
            # Update filters
            filter_state = torch.einsum('bf,fg->bg', filter_state, self.filter_update)
            output[timestep + 1] = filter_state
        
        # Weight the filterbanks
        w = self.op.weight
        weighted_filters = w.view(1, *w.shape) * output.view(timesteps + 1, 
                                                             1,
                                                             self.src_shape * self.nb_filters)
        
        weighted_filters = weighted_filters.reshape(timesteps + 1,
                                                    w.shape[0],
                                                    self.src_shape,
                                                    self.nb_filters)
        
        return weighted_filters.cpu().detach().numpy()
    
        
class IdentityConnection(BaseConnection):
    def __init__(self, src, dst, target=None, bias=False, requires_grad=True, name=None, regularizers=None, constraints=None, tie_weights=None, weight_scale=1.0):
        """ Initialize IdentityConnection

        Args:
            tie_weights (list of int, optional): Tie weights along dims given in list
            weight_scale (float, optional): Scale everything by this factor. Useful when the connection is used for relaying currents rather than spikes.
        """
        super(IdentityConnection, self).__init__(src, dst, name=name,
                                                 target=target, regularizers=regularizers, constraints=constraints)

        self.requires_grad = requires_grad
        self.weight_scale = weight_scale
        wshp = src.shape

        # Set weights tensor dimension to 1 along tied dimensions
        if tie_weights is not None:
            wshp = list(wshp)
            for d in tie_weights:
                wshp[d] = 1
            wshp = tuple(wshp)

        self.weights = Parameter(torch.randn(
            wshp), requires_grad=requires_grad)
        if bias:
            self.bias = Parameter(torch.randn(
                wshp), requires_grad=requires_grad)

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
                self.weight_scale, self.target, self.weights * preact)
        else:
            self.dst.scale_and_add_to_state(
                self.weight_scale, self.target, self.weights * preact + self.bias)

    def propagate(self):
        self.forward()


class ConvConnection(Connection):
    def __init__(self, src, dst, conv=nn.Conv1d, **kwargs):
        super(ConvConnection, self).__init__(
            src, dst, operation=conv, **kwargs)


class Conv2dConnection(Connection):
    def __init__(self, src, dst, conv=nn.Conv2d, **kwargs):
        super(Conv2dConnection, self).__init__(
            src, dst, operation=conv, **kwargs)


class SuperConvConnection(SuperConnection):
    def __init__(self, src, dst, conv=nn.Conv1d, **kwargs):
        super(SuperConvConnection, self).__init__(
            src, dst, operation=conv, **kwargs)
        
    def forward(self):
        
        preact = self.src.out
        
        # Update filters
        new_filters = torch.einsum('bncf,fg->bncg', self.filters, self.filter_update)
        
        # add spiketrain to first filters
        new_filters[:, :, :, 0] += preact

        self.filters = new_filters
                    
        filter_out = new_filters.view(self.filters.shape[0], 
                                      self.src_shape * self.nb_filters,
                                      -1)
        
        if not self.propagate_gradients:
            filter_out = filter_out.detach()
        
        if self.flatten_input:
            shp = filter_out.shape
            filter_out = filter_out.reshape(shp[:1] + (-1,))

        out = self.op(filter_out)
        self.dst.add_to_state(self.target, out)

    def propagate(self):
        self.forward()

    def apply_constraints(self):
        for const in self.constraints:
            const.apply(self.op.weight)
            
    def get_filterbanks(self, T=1):
        """
        Used to inspect the learned synaptic filters.
        Returns an object of shape (time, n_pre, n_post, n_filters, kernel_size)
        """
        
        # Prepare empty tensors
        timesteps = int(T / self.time_step)
        
        filter_shape = (self.src_shape, self.nb_filters)
        filter_state = torch.zeros(filter_shape, device=self.device, dtype=self.dtype)
        output = torch.zeros((timesteps + 1, *filter_shape), device=self.device, dtype=self.dtype)

        # Evolve filters over time
        filter_state[..., 0] = 1
        output[0] = filter_state
        
        for timestep in range(timesteps):
            
            # Update filters
            filter_state = torch.einsum('nf,fg->ng', filter_state, self.filter_update)
            output[timestep + 1] = filter_state
        
        # Weight the filterbanks
        w = self.op.weight
        weighted_filters = w.view(1, *w.shape) * output.view(timesteps + 1, 
                                                             1,
                                                             self.src_shape * self.nb_filters,
                                                             1)
        
        weighted_filters = weighted_filters.reshape(timesteps + 1,
                                                    w.shape[0],
                                                    self.src_shape,
                                                    self.nb_filters,
                                                    -1)
        
        return weighted_filters.cpu().detach().numpy()