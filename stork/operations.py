from torch.nn.modules import Linear
from torch.nn import functional as F

class MaskedLinear(Linear):

    def forward(self, input, mask=1):
        return F.linear(input, self.weight * mask, self.bias)