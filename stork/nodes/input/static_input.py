from stork.nodes.input.base import InputGroup


class StaticInputGroup(InputGroup):
    """A special group which is used to supply batched dense tensor input to the network via its feed_data function."""

    def __init__(self, shape, scale=1, name="Input"):
        super(StaticInputGroup, self).__init__(shape, name=name)
        self.scale = scale

    def feed_data(self, data):
        self.local_data = data.reshape(((data.shape[0],) + self.shape)).to(self.device)

    def forward(self):
        self.out = self.states["out"] = self.scale * self.local_data
