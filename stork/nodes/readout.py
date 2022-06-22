import numpy as np
import torch

from stork.nodes.base import CellGroup


class ReadoutGroup(CellGroup):
    def __init__(self, shape, tau_mem=10e-3, tau_syn=5e-3, weight_scale=1.0, initial_state=-1e-3, stateful=False):
        super().__init__(shape, stateful=stateful, name="Readout")
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.store_output_seq = True
        self.initial_state = initial_state
        self.weight_scale = weight_scale
        self.out = None
        self.syn = None

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)
        self.dcy_mem = float(np.exp(-time_step / self.tau_mem))
        self.scl_mem = 1.0 - self.dcy_mem
        self.dcy_syn = float(np.exp(-time_step / self.tau_syn))
        self.scl_syn = (1.0 - self.dcy_syn) * self.weight_scale

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.out = self.get_state_tensor("out", state=self.out, init=self.initial_state)
        self.syn = self.get_state_tensor("syn", state=self.syn)

    def forward(self):
        # synaptic & membrane dynamics
        new_syn = self.dcy_syn * self.syn + self.input
        new_mem = self.dcy_mem * self.out + self.scl_mem * self.syn

        self.out = self.states["out"] = new_mem
        self.syn = self.states["syn"] = new_syn
        # self.out_seq.append(self.out)


