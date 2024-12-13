import torch.nn as nn


class NetworkNode(nn.Module):
    def __init__(self, name=None, regularizers=None):
        """Initialize base class

        Args:
            name: A string name for this class used in logs
            regularizers: A list of regularizers for this class
        """
        super(NetworkNode, self).__init__()
        if name is None:
            self.name = ""
        else:
            self.name = name

        if regularizers is None:
            self.regularizers = []
        else:
            self.regularizers = regularizers

    def set_name(self, name):
        self.name = name

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.time_step = time_step
        self.device = device
        self.dtype = dtype
        
    def set_nb_steps(self, nb_steps):
        self.nb_steps = nb_steps

    def remove_regularizers(self):
        self.regularizers = []

    def set_dtype(self, dtype):
        self.dtype = dtype