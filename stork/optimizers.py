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
        """ Performs a single gradient step. """
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:  # skip if param has no gradient
                    continue
                grad = p.grad.data
                param_state = self.state[p]
                if 'mem' not in param_state:  # setup accumulators once
                    param_state['mem'] = torch.full_like(p.data, 1.)
                    param_state['g'] = torch.full_like(p.data, 0.)
                    param_state['g2'] = torch.full_like(p.data, 0.)
                mem = param_state['mem']
                g, g2 = param_state['g'], param_state['g2']
                r = 1. / (mem + 1.)
                g.mul_(1 - r).addcmul_(r, grad)
                g2.mul_(1 - r).addcmul_(r, grad**2)
                div = g2 + self.eps
                mem.mul_(1 - (g**2 / div)).add_(1.)
                lrate = torch.clamp(g**2 / div, max=lr)
                new_grad = -lrate*grad / (g2.sqrt() + self.eps)
                p.data.add_(new_grad)
        return loss


class SMORMS4(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, eps=1e-16, min_mem_decay=0.5):
        """
        SMORMS4 optimizer. A modfied SMORMS3 for improved stability and performance.

        fzenke, 2021-08-18

        Args
            lr: learning rate
            eps: epsilon
            min_mem_decay: Minimum decay rate for moment estimation.
        """
        defaults = dict(lr=lr)
        super(SMORMS4, self).__init__(params, defaults)
        self.eps = eps
        self.min_mem_decay = min_mem_decay
   
    def step(self, closure=None):
        """ Performs a single gradient step. """
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
             if p.grad is None: continue # skip if param has no gradient
             grad = p.grad.data
             param_state = self.state[p]
             if 'mem' not in param_state: # setup accumulators once
                 param_state['mem'] = torch.full_like(p.data, 1.)
                 param_state['g'] = torch.full_like(p.data, 0.)
                 param_state['g2'] = torch.full_like(p.data, 0.)
             mem = param_state['mem']
             g, g2 = param_state['g'], param_state['g2']
             # gsquare = torch.square(torch.clamp(grad,min=-1e18,max=1e18))
             gsquare = torch.square(grad)
             r = 1.0 / (mem + 1.0)
             g.mul_(1 - r).addcmul_(r, grad)
             # g2.mul_(1 - r).addcmul_(r, gsquare)
             g2 = param_state['g2'] = torch.maximum( (1 - r) * g2 , gsquare )
             div = g2 + self.eps
             xi = ( g**2 / div ).sqrt()
             mem.mul_(torch.clamp( 1.0 - xi, min=self.min_mem_decay, max=1.0)).add_(1.) 
             lrate = torch.clamp( xi, max=lr )
             new_grad = -lrate * grad / ( g2.sqrt() + self.eps )
             p.data.add_( new_grad )
        return loss