import torch
from torch import nn as nn

import numpy as np

from stork import core


class CellGroup(core.NetworkNode):
    """
    Base class from which all neurons are derived.

    """
    clk = 0

    def __init__(self, shape, store_sequences=None, name=None, regularizers=None, dropout_p=0.0, stateful=False):
        super(CellGroup, self).__init__(name, regularizers)
        if type(shape) == int:
            self.shape = (shape,)
        else:
            self.shape = tuple(shape)
        self.nb_units = int(np.prod(self.shape))
        self.states = {}
        self.store_state_sequences = ["out"]
        if store_sequences is not None:
            self.store_state_sequences.extend(store_sequences)
            self.store_state_sequences = list(set(self.store_state_sequences))
        self.stored_sequences_ = {}
        self.default_target = "input"
        self.stateful = stateful
        if dropout_p:
            self.dropout = nn.Dropout(dropout_p)
        else:
            self.dropout = None

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)
        self.reset_state()

    def get_regularizer_loss(self):
        reg_loss = torch.tensor(0.0, device=self.device)
        for reg in self.regularizers:
            reg_loss += reg(self)
        return reg_loss

    def set_state_tensor(self, key, state):
        self.states[key] = state

    def prepare_state_tensor_(self, state=None, init=0.0, shape=None):
        """ Prepares a state tensor by either initializing it or copying the previous one.

        Args: 
            state (tensor): The previous state tensor if one exists
            init (float): Numerical value to init tensor with
            shape (None or tuple): Shape of the state. Assuming a single value if none.

        Returns:
            A tensor with dimensions current_batch_size x neuronal_shape x shape 
        """

        if self.stateful and state is not None and state.size() == self.int_shape:
            new_state = state.detach()
        else:
            if shape is None:
                full_shape = self.int_shape
            else:
                full_shape = self.int_shape+shape

            if init:
                new_state = init * \
                    torch.ones(full_shape, device=self.device,
                               dtype=self.dtype)
            else:
                new_state = torch.zeros(
                    full_shape, device=self.device, dtype=self.dtype)

        return new_state

    def get_state_tensor(self, key, state=None, init=0.0, shape=None):
        self.states[key] = state = self.prepare_state_tensor_(
            state=state, init=init, shape=shape)
        return state

    def add_to_state(self, target, x):
        """ Add x to state tensor. Mostly used by Connection objects to implement synaptic transmission. """
        self.states[target] += x

    def scale_and_add_to_state(self, scale, target, x):
        """ Add x to state tensor. Mostly used by Connection objects to implement synaptic transmission. """
        self.add_to_state(target, scale*x)

    def clear_input(self):
        self.input = self.states["input"] = torch.zeros(
            self.int_shape, device=self.device, dtype=self.dtype)

    def reset_state(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        self.int_shape = (batch_size,) + self.shape
        self.flat_seq_shape = (batch_size, self.nb_steps, self.nb_units)
        self.clear_input()
        for key in self.store_state_sequences:
            self.stored_sequences_[key] = []

    def evolve(self):
        """ Advances simulation of group by one timestep and append output to out_seq. """
        self.forward()
        self.set_state_tensor("out", self.out)
        if self.dropout is not None:
            self.out = self.dropout(self.out)
        for key in self.store_state_sequences:
            self.stored_sequences_[key].append(self.states[key])

    def get_state_sequence(self, key):
        seq = self.stored_sequences_[key]
        # if this a list of states, concatenate it along time dimension and store result as tensor for caching
        if type(seq) == list:
            seq = self.stored_sequences_[key] = torch.stack(seq, dim=1)
        return seq

    def get_in_sequence(self):
        if "input" in self.store_state_sequences:
            return self.get_state_sequence("input")
        else:
            print(
                "Warning requested input sequence was not stored. Add 'input' to  store_state_sequences list.")
            return None

    def get_out_sequence(self):
        return self.get_state_sequence("out")

    def get_flattened_out_sequence(self):
        return self.get_state_sequence("out").reshape(self.flat_seq_shape)

    def get_firing_rates(self):
        tmp = self.get_out_sequence()
        rates = torch.mean(tmp, dim=1) / self.time_step  # Average over time
        return rates

    def get_mean_population_rate(self):
        rate = torch.mean(self.get_firing_rates())
        return rate

    def get_out_channels(self):
        return self.shape[0]

    def __call__(self, inputs):
        raise NotImplementedError
