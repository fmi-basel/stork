import torch
from torch.nn.modules import Linear
from torch.nn import functional as F


class MaskedLinear(Linear):
    def forward(self, input, mask=1):
        # TODO: add bias
        w = self.weight.unsqueeze(0)
        mw = w * mask
        mwT = torch.transpose(mw, 1, 2)

        return input.unsqueeze(1) @ mwT
