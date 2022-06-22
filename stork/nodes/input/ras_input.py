import numpy as np
import torch

from stork.nodes.input.base import InputGroup


class RasInputGroup(InputGroup):
    """ Like InputGroup but eats ras format instead of dense tensors. """

    def feed_data(self, data):
        super().feed_data(data)

    def forward(self):
        tmp = torch.zeros(self.int_shape, dtype=self.dtype)
        for bi, dat in enumerate(self.local_data):
            times, units = dat
            idx = np.array(units[times == self.clk], dtype=np.int)
            tmp[bi, idx] = 1.0
        self.out = tmp.to(device=self.device)