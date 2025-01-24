import torch


class SMORMS3(torch.optim.Optimizer):
    """
    Optimizer as described by Simon Funk
    https://sifter.org/~simon/journal/20150420.html
    """

    def __init__(self, params, lr=1e-3, eps=1e-16):
        """
        Setup optimizer with parameters.
        Args
            lr (float): learning rate. Defaults to 1e-3.
            eps (float): epsilon. Defaults to 1e-16
        """
        defaults = dict(lr=lr)
        super(SMORMS3, self).__init__(params, defaults)
        self.eps = eps

    def step(self, closure=None):
        """Performs a single gradient step."""
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:  # skip if param has no gradient
                    continue
                grad = p.grad.data
                param_state = self.state[p]
                if "mem" not in param_state:  # setup accumulators once
                    param_state["mem"] = torch.full_like(p.data, 1.0)
                    param_state["g"] = torch.full_like(p.data, 0.0)
                    param_state["g2"] = torch.full_like(p.data, 0.0)
                mem = param_state["mem"]
                g, g2 = param_state["g"], param_state["g2"]
                r = 1.0 / (mem + 1.0)
                g.mul_(1 - r).addcmul_(r, grad)
                g2.mul_(1 - r).addcmul_(r, grad**2)
                div = g2 + self.eps
                mem.mul_(1 - (g**2 / div)).add_(1.0)
                lrate = torch.clamp(g**2 / div, max=lr)
                new_grad = -lrate * grad / (g2.sqrt() + self.eps)
                p.data.add_(new_grad)
        return loss
