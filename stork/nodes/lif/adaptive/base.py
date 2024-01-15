import numpy as np

from stork.nodes.lif.base import LIFGroup


class AdaptiveLIFGroup(LIFGroup):
    """
    Base class for LIF neurons with adaptive threshold

    Args:
        shape: The neuron group shape
        tau_ada: The adaptation time constant
        adapt_a: The adaptation strength
    """

    def __init__(self, shape, tau_ada=100e-3, adapt_a=0.1, **kwargs):
        super().__init__(shape, **kwargs)
        self.tau_ada = tau_ada
        self.adapt_a = adapt_a
        self.ada = None

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)
        self.dcy_ada = float(np.exp(-time_step / self.tau_ada))
        self.scl_ada = 1.0 - self.dcy_ada

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.ada = self.get_state_tensor("ada", state=self.ada)

    def forward(self):
        # spike & reset
        new_out, rst = self.get_spike_and_reset(self.mem)

        # synaptic & membrane dynamics
        new_syn = self.dcy_syn * self.syn + self.input
        new_mem = (
            self.dcy_mem * self.mem
            + self.scl_mem * (self.syn - self.adapt_a * self.ada)
        ) * (1.0 - rst)
        new_ada = self.dcy_ada * self.ada + self.out

        self.out = self.states["out"] = new_out
        self.mem = self.states["mem"] = new_mem
        self.syn = self.states["syn"] = new_syn
        self.ada = self.states["ada"] = new_ada
