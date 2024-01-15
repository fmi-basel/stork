import torch

from stork import activations
from stork.nodes.base import CellGroup


class ExcInhLIFGroup(CellGroup):
    def __init__(
        self,
        shape,
        tau_mem=10e-3,
        tau_exc=5e-3,
        tau_inh=5e-3,
        sigma_tau=0.0,
        activation=activations.SuperSpike,
        **kwargs
    ):
        super(ExcInhLIFGroup, self).__init__(shape, **kwargs)
        self.spk_nl = activation.apply
        self.tau_mem = tau_mem
        self.tau_exc = tau_exc
        self.tau_inh = tau_inh
        self.thr = 1.0
        self.sigma_tau = sigma_tau

        self.mem = None
        self.out = None
        self.exc = None
        self.inh = None
        self.syne = None
        self.syni = None
        self.default_target = "exc"

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)
        tau_mem = torch.tensor(self.tau_mem, dtype=dtype).to(device)
        tau_exc = torch.tensor(self.tau_exc, dtype=dtype).to(device)
        tau_inh = torch.tensor(self.tau_inh, dtype=dtype).to(device)
        if self.sigma_tau:
            tau_mem, tau_exc, tau_inh = [
                tau
                * torch.exp(
                    self.sigma_tau * torch.randn(self.shape, dtype=dtype).to(device)
                )
                for tau in [tau_mem, tau_exc, tau_inh]
            ]
        self.dcy_mem = torch.exp(-time_step / tau_mem)
        self.dcy_exc = torch.exp(-time_step / tau_exc)
        self.dcy_inh = torch.exp(-time_step / tau_inh)
        self.scl_mem = 1.0 - self.dcy_mem
        self.scl_exc = 1.0 - self.dcy_exc
        self.scl_inh = 1.0 - self.dcy_inh

    def clear_input(self):
        # no prev state avoids stateful
        self.exc = self.get_state_tensor("exc")
        self.inh = self.get_state_tensor("inh")

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.mem = self.get_state_tensor("mem", state=self.mem)
        self.out = self.get_state_tensor("out", state=self.out)
        self.syne = self.get_state_tensor("syne", state=self.syne)
        self.syni = self.get_state_tensor("syni", state=self.syni)

    def get_spike_and_reset(self, mem):
        out = self.spk_nl(mem - self.thr)
        rst = out.detach()
        return out, rst

    def forward(self):
        # spike & reset
        new_out, rst = self.get_spike_and_reset(self.mem)

        # synaptic & membrane dynamics
        new_syne = self.dcy_exc * self.syne + self.exc
        new_syni = self.dcy_inh * self.syni + self.inh
        net_input_current = self.syne - self.syni
        self.set_state_tensor("net_input_current", net_input_current)
        new_mem = (self.dcy_mem * self.mem + self.scl_mem * (net_input_current)) * (
            1.0 - rst
        )

        self.out = self.states["out"] = new_out
        self.mem = self.states["mem"] = new_mem
        self.syne = self.states["syne"] = new_syne
        self.syni = self.states["syni"] = new_syni


class ExcInhAdaptiveLIFGroup(CellGroup):
    def __init__(
        self,
        shape,
        tau_mem=10e-3,
        tau_exc=5e-3,
        tau_inh=5e-3,
        tau_adapt=100e-3,
        adapt_a=0.5,
        sigma_tau=0.0,
        **kwargs
    ):
        super(ExcInhAdaptiveLIFGroup, self).__init__(shape, **kwargs)
        self.spk_nl = activations.SuperSpike.apply
        self.tau_mem = tau_mem
        self.tau_exc = tau_exc
        self.tau_inh = tau_inh
        self.thr = 1.0
        self.sigma_tau = sigma_tau
        self.tau_ada = tau_adapt
        self.adapt_a = adapt_a

        self.mem = None
        self.out = None
        self.exc = None
        self.inh = None
        self.ada = None
        self.syne = None
        self.syni = None
        self.default_target = "exc"

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)
        tau_mem = torch.tensor(self.tau_mem, dtype=dtype).to(device)
        tau_exc = torch.tensor(self.tau_exc, dtype=dtype).to(device)
        tau_inh = torch.tensor(self.tau_inh, dtype=dtype).to(device)
        tau_ada = torch.tensor(self.tau_ada, dtype=dtype).to(device)
        if self.sigma_tau:
            tau_mem, tau_exc, tau_inh, tau_ada = [
                tau
                * torch.exp(
                    self.sigma_tau * torch.randn(self.shape, dtype=dtype).to(device)
                )
                for tau in [tau_mem, tau_exc, tau_inh, tau_ada]
            ]
        self.dcy_mem = torch.exp(-time_step / tau_mem)
        self.dcy_exc = torch.exp(-time_step / tau_exc)
        self.dcy_inh = torch.exp(-time_step / tau_inh)
        self.dcy_ada = torch.exp(-time_step / tau_ada)
        self.scl_mem = 1.0 - self.dcy_mem
        self.scl_exc = 1.0 - self.dcy_exc
        self.scl_inh = 1.0 - self.dcy_inh
        self.scl_ada = 1.0 - self.dcy_ada

    def clear_input(self):
        # no prev state avoids stateful
        self.exc = self.get_state_tensor("exc")
        self.inh = self.get_state_tensor("inh")

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.mem = self.get_state_tensor("mem", state=self.mem)
        self.out = self.get_state_tensor("out", state=self.out)
        self.ada = self.get_state_tensor("out", state=self.ada)
        self.syne = self.get_state_tensor("syne", state=self.syne)
        self.syni = self.get_state_tensor("syni", state=self.syni)

    def get_spike_and_reset(self, mem):
        out = self.spk_nl(mem - self.thr)
        rst = out.detach()
        return out, rst

    def forward(self):
        # spike & reset
        new_out, rst = self.get_spike_and_reset(self.mem)

        # synaptic & membrane dynamics
        new_syne = self.dcy_exc * self.syne + self.exc
        new_syni = self.dcy_inh * self.syni + self.inh
        net_input_current = self.syne - self.syni
        new_ada = self.dcy_ada * self.ada + self.out
        self.set_state_tensor("net_input_current", net_input_current)
        new_mem = (
            self.dcy_mem * self.mem
            + self.scl_mem * (net_input_current - self.adapt_a * self.ada)
        ) * (1.0 - rst)

        self.out = self.states["out"] = new_out
        self.mem = self.states["mem"] = new_mem
        self.ada = self.states["ada"] = new_ada
        self.syne = self.states["syne"] = new_syne
        self.syni = self.states["syni"] = new_syni


class Exc2InhLIFGroup(CellGroup):
    def __init__(
        self,
        shape,
        tau_mem=10e-3,
        tau_ampa=5e-3,
        tau_nmda=100e-3,
        tau_gaba=10e-3,
        sigma_tau=0.0,
        **kwargs
    ):
        super(Exc2InhLIFGroup, self).__init__(shape, **kwargs)
        self.spk_nl = activations.SuperSpike.apply
        self.tau_mem = tau_mem
        self.tau_ampa = tau_ampa
        self.tau_gaba = tau_gaba
        self.tau_nmda = tau_nmda
        self.thr = 1.0
        self.sigma_tau = sigma_tau

        self.out = None
        self.mem = None
        self.ampa = None
        self.gaba = None
        self.nmda = None
        self.exc = None
        self.inh = None
        self.default_target = "exc"

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)
        tau_mem = torch.tensor(self.tau_mem, dtype=dtype).to(device)
        tau_ampa = torch.tensor(self.tau_ampa, dtype=dtype).to(device)
        tau_nmda = torch.tensor(self.tau_nmda, dtype=dtype).to(device)
        tau_gaba = torch.tensor(self.tau_gaba, dtype=dtype).to(device)
        if self.sigma_tau:
            tau_mem, tau_ampa, tau_nmda, tau_gaba = [
                tau
                * torch.exp(
                    self.sigma_tau * torch.randn(self.shape, dtype=dtype).to(device)
                )
                for tau in [tau_mem, tau_ampa, tau_nmda, tau_gaba]
            ]
        self.dcy_mem = torch.exp(-time_step / tau_mem)
        self.dcy_ampa = torch.exp(-time_step / tau_ampa)
        self.dcy_nmda = torch.exp(-time_step / tau_nmda)
        self.dcy_gaba = torch.exp(-time_step / tau_gaba)
        self.scl_mem = 1.0 - self.dcy_mem
        self.scl_ampa = 1.0 - self.dcy_ampa
        self.scl_nmda = 1.0 - self.dcy_nmda
        self.scl_gaba = 1.0 - self.dcy_gaba

    def clear_input(self):
        # not passing previous state forces init during stateful
        self.exc = self.get_state_tensor("exc")
        self.inh = self.get_state_tensor("inh")
        self.net_input_current = self.get_state_tensor(
            "net_input_current", state=torch.zeros_like(self.inh)
        )

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.out = self.get_state_tensor("out", state=self.out)
        self.mem = self.get_state_tensor("mem", state=self.mem)
        self.ampa = self.get_state_tensor("ampa", state=self.ampa)
        self.gaba = self.get_state_tensor("gaba", state=self.gaba)
        self.nmda = self.get_state_tensor("nmda", state=self.nmda)

    def get_spike_and_reset(self, mem):
        out = self.spk_nl(mem - self.thr)
        rst = out.detach()
        return out, rst

    def forward(self):
        # spike & reset
        new_out, rst = self.get_spike_and_reset(self.mem)

        # synaptic & membrane dynamics
        new_ampa = self.dcy_ampa * self.ampa + self.exc
        new_gaba = self.dcy_gaba * self.gaba + self.inh
        new_nmda = self.dcy_nmda * self.nmda + self.scl_nmda * self.ampa
        net_input_current = (self.ampa + self.nmda) / 2 - self.gaba
        self.set_state_tensor("net_input_current", net_input_current)
        new_mem = (self.dcy_mem * self.mem + self.scl_mem * (net_input_current)) * (
            1.0 - rst
        )

        self.out = self.states["out"] = new_out
        self.mem = self.states["mem"] = new_mem
        self.ampa = self.states["ampa"] = new_ampa
        self.gaba = self.states["gaba"] = new_gaba
        self.nmda = self.states["nmda"] = new_nmda
