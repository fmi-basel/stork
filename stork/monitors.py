import torch


class Monitor:
    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedError


class SpikeMonitor(Monitor):
    """Records spikes in sparse RAS format

    Args:
        group: The group to record from

    Returns:
        argwhere of out sequence
    """

    def __init__(self, group):
        super().__init__()
        self.group = group
        self.batch_count_ = 0

    def reset(self):
        pass

    def execute(self):
        pass

    def get_data(self):
        out = self.group.get_out_sequence().detach().cpu()
        tmp = torch.nonzero(out)
        tmp[:, 0] += self.batch_count_
        self.batch_count_ += out.shape[0]
        return tmp


class StateMonitor(Monitor):
    """Records the state of a neuron group over time

    Args:
        group: The group to record from
        key: The name of the state
    """

    def __init__(self, group, key, subset=None):
        super().__init__()
        self.group = group
        self.key = key
        self.subset = subset

    def reset(self):
        self.data = []

    def execute(self):
        if self.subset is not None:
            self.data.append(self.group.states[self.key][:, self.subset].detach().cpu())
        else:
            self.data.append(self.group.states[self.key].detach().cpu())

    def get_data(self):
        return torch.stack(self.data, dim=1)


class SpikeCountMonitor(Monitor):
    """Counts number of spikes (sum over time in get_out_sequence() for each neuron)

    Args:
        group: The group to record from

    Returns:
        A tensor with spike counts for each input and neuron
    """

    def __init__(self, group):
        super().__init__()
        self.group = group

    def reset(self):
        pass

    def execute(self):
        pass

    def get_data(self):
        return torch.sum(self.group.get_out_sequence().detach().cpu(), dim=1)


class PopulationSpikeCountMonitor(Monitor):
    """Counts total number of spikes (sum over time in get_out_sequence() for the group)

    Args:
        group: The group to record from

    Returns:
        A tensor with spike counts for each input and neuron
    """

    def __init__(self, group):
        super().__init__()
        self.group = group

    def reset(self):
        self.data = []

    def execute(self):
        pass

    def get_data(self):
        s1 = torch.sum(self.group.get_out_sequence().detach().cpu(), dim=1)
        return torch.mean(s1)


class PopulationFiringRateMonitor(Monitor):
    """Monitors population firing rate (nr of spikes / nr of neurons for every timestep)

    Args:
        group: The group to record from

    Returns:
        A tensor with population firing rate for each input and timestep
    """

    def __init__(self, group):
        super().__init__()
        self.group = group

    def reset(self):
        self.data = []

    def execute(self):
        pass

    def get_data(self):
        s1 = self.group.get_out_sequence().detach().cpu()
        s1 = s1.reshape(s1.shape[0], s1.shape[1], self.group.nb_units)
        return torch.sum(s1, dim=-1) / self.group.nb_units


class MeanVarianceMonitor(Monitor):
    """Measures mean and variance of input

    Args:
        group: The group to record from
        state (string): State variable to monitor (Monitors mean and variance of a state variable)


    Returns:
        A tensors with mean and variance for each neuron/state along the last dim
    """

    def __init__(self, group, state="input"):
        super().__init__()
        self.group = group
        self.key = state

    def reset(self):
        self.s = 0
        self.s2 = 0
        self.c = 0

    def execute(self):
        tns = self.group.states[self.key]
        self.s += tns.detach().cpu()
        self.s2 += torch.square(tns).detach().cpu()
        self.c += 1

    def get_data(self):
        mean = self.s / self.c
        var = self.s2 / self.c - mean**2
        return torch.stack((mean, var), len(mean.shape))


class GradientMonitor(Monitor):
    """Records the gradients (weight.grad)

    Args:
        target: The tensor or nn.Module to record from
                (usually a stork.connection.op object)
                Needs to have a .weight argument
    """

    def __init__(self, target):
        super().__init__()
        self.target = target

    def reset(self):
        pass

    def set_hook(self):
        """
        Sets the backward hook
        """
        pass

    def remove_hook(self):
        pass

    def execute(self):
        pass

    def get_data(self):
        # unsqueeze so that the output from the monitor is [batch_nr x weightmatrix-dims]
        return self.target.weight.grad.detach().cpu().abs().unsqueeze(0)


class GradientOutputMonitor(GradientMonitor):
    """Records the gradients wrt the neuronal output
        computed in the backward pass

    Args:
        target: The tensor or nn.Module to record from
                (usually a stork.connection.op object)
    """

    def __init__(self, target):
        super().__init__(target)
        self.count = 0
        self.sum = 0

    def set_hook(self):
        """
        Sets the backward hook
        """
        self.hook = self.target.register_full_backward_hook(self.grab_gradient)

    def remove_hook(self):
        self.hook.remove()

    def grab_gradient(self, module, grad_input, grad_output):
        mean_grad = grad_output[0].detach().cpu().abs()
        self.sum += mean_grad
        self.count += 1

    def reset(self):
        self.count = 0
        self.sum = 0

    def execute(self):
        pass

    def get_data(self):
        return self.sum / self.count
