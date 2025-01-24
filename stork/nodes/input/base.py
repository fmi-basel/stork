import torch

from stork.nodes.base import CellGroup


class InputGroup(CellGroup):
    """A special group which is used to supply batched dense tensor input to the network via its feed_data function."""

    def __init__(self, shape, name="Input", **kwargs):
        super(InputGroup, self).__init__(shape, name=name, **kwargs)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.out = self.states["out"] = torch.zeros(self.int_shape, device=self.device, dtype=self.dtype)

    def feed_data(self, data):
        self.local_data = data.reshape((data.shape[:2] + self.shape)).to(self.device)

    def forward(self):
        self.out = self.states["out"] = self.local_data[:, self.clk]