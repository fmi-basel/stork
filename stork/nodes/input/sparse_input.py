import torch

from stork.nodes.input.base import InputGroup


class SparseInputGroup(InputGroup):
    """ Like InputGroup but eats sparse tensors instead of dense ones. """

    def __init__(self, shape):
        super(InputGroup, self).__init__(shape)

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.out = torch.zeros(self.int_shape, device=self.device, dtype=self.dtype)

    def feed_data(self, data):
        self.local_data = data

    def forward(self):
        i, v = self.local_data[self.clk]
        tmp = torch.sparse.FloatTensor(i, v, torch.Size(self.int_shape))
        # print(self.int_shape)
        self.out = tmp.to(self.device)