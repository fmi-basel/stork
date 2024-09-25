import numpy as np
import torch
from torch.nn import Parameter

from stork import activations
from stork.nodes.base import CellGroup


class LIFGroup(CellGroup):
    def __init__(self,
                 shape,
                 tau_mem=10e-3,
                 tau_syn=5e-3,
                 diff_reset=False,
                 learn_tau_mem=False,
                 learn_tau_syn=False,
                 learn_tau_mem_hetero=False,
                 learn_tau_syn_hetero=False,
                 clamp_mem=False,
                 activation=activations.SuperSpike,
                 dropout_p=0.0,
                 stateful=False,
                 name="LIFGroup",
                 regularizers=None,
                 **kwargs):
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

        super().__init__(shape, dropout_p=dropout_p, stateful=stateful,
                         name=name, regularizers=regularizers, **kwargs)
        
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
        assert self.tau_mem.shape == torch.Size([]) or self.tau_mem.shape == torch.Size([shape])
        assert self.tau_syn.shape == torch.Size([]) or self.tau_syn.shape == torch.Size([shape])
         
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
        
    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        
        # Set device and dtype for tau_mem and tau_syn
        self.tau_mem = self.tau_mem.to(device=device, dtype=dtype)
        self.tau_syn = self.tau_syn.to(device=device, dtype=dtype)
        self.to(device=device, dtype=dtype)     # This moves parameters
        
        self.dcy_mem = torch.exp(-time_step / self.tau_mem)
        self.scl_mem = 1.0 - self.dcy_mem
        self.dcy_syn = torch.exp(-time_step / self.tau_syn)
        self.scl_syn = 1.0 - self.dcy_syn
        
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        
        if self.learn_tau_mem:
            time_step = torch.tensor(self.time_step, device=self.device, dtype=self.dtype)
            self.dcy_mem = torch.exp(-time_step / (self.tau_mem * torch.nn.functional.softplus(self.mem_param)))
            self.scl_mem = 1.0 - self.dcy_mem
            
        if self.learn_tau_syn:
            if 'time_step' not in locals():
                time_step = torch.tensor(self.time_step, device=self.device, dtype=self.dtype)
            self.dcy_syn = torch.exp(-time_step / (self.tau_syn * torch.nn.functional.softplus(self.syn_param)))
            self.scl_syn = 1.0 - self.dcy_syn
            
        self.mem = self.get_state_tensor("mem", state=self.mem)
        self.syn = self.get_state_tensor("syn", state=self.syn)
        self.out = self.states["out"] = torch.zeros(self.int_shape, device=self.device, dtype=self.dtype)

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
        new_mem = (self.dcy_mem * self.mem + self.scl_mem * self.syn) * (1.0 - rst)  # multiplicative reset

        # Clamp membrane potential
        if self.clamp_mem:
            new_mem = torch.clamp(new_mem, max=1.01)

        self.out = self.states["out"] = new_out
        self.mem = self.states["mem"] = new_mem
        self.syn = self.states["syn"] = new_syn
