import torch
from torch.nn import Parameter

from stork.nodes.lif.base import LIFGroup


class AdaptLearnLIFGroup(LIFGroup):
    def __init__(self, nb_units, tau_ada=1.0, **kwargs):
        super().__init__(nb_units, **kwargs)
        self.tau_ada = tau_ada
        self.ada = None

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        self.dt_ = torch.tensor(time_step, device=device, dtype=dtype)

        adapt_b = torch.rand(self.shape, device=device, dtype=dtype, requires_grad=True)
        self.adapt_b = Parameter(adapt_b, requires_grad=True)

        ada_dcy_param = torch.randn(
            self.shape, device=device, dtype=dtype, requires_grad=True
        )
        self.ada_dcy_param = Parameter(ada_dcy_param, requires_grad=True)
        # Reset state is invoked by configure
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.ada = self.get_state_tensor("ada", state=self.ada)
        self.dcy_ada = torch.exp(
            -self.dt_ / (torch.sigmoid(self.ada_dcy_param) * self.tau_ada)
        )

    def forward(self):
        # spike & reset
        new_out, rst = self.get_spike_and_reset(self.mem)

        # synaptic & membrane dynamics
        new_syn = self.dcy_syn * self.syn + self.input
        new_mem = (
            self.dcy_mem * self.mem
            + self.scl_mem * (self.syn - self.adapt_b * self.ada)
        ) * (1.0 - rst)

        # decay of adaptation variable
        new_ada = self.dcy_ada * self.ada + (1.0 - self.dcy_ada) * (self.out)

        self.out = self.states["out"] = new_out
        self.mem = self.states["mem"] = new_mem
        self.syn = self.states["syn"] = new_syn
        self.ada = self.states["ada"] = new_ada
