import torch
from torch.nn.modules import Linear
from torch.nn import functional as F


masked_linear = torch.vmap(F.linear, in_dims=(0, 0, None))
class MaskedLinear(Linear):
    def forward(self, input, mask=1):
        return masked_linear(input, self.weight.unsqueeze(0) * mask, self.bias)
