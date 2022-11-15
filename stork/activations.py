import torch
import torch.nn as nn


class SuperSpike(torch.autograd.Function):
    """
    Autograd SuperSpike nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta.
    """
    beta = 20.0

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations. 
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SuperSpike.beta*torch.abs(input)+1.0)**2
        return grad


class SuperSpike_MemClamp(torch.autograd.Function):
    """
    Variant of SuperSpike with clamped membrane potential at 1.0
    """
    beta = 20.0

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations. 
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SuperSpike_MemClamp.beta *
                           torch.abs(torch.relu(-input))+1.0)**2
        return grad


class SuperSpike_rescaled(torch.autograd.Function):
    """
    Version of SuperSpike where the gradient is re-scaled so that it equals one at 
    resting membrane potential
    """
    beta = 20.0

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations. 
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        rescale_val = 1 / ((SuperSpike_rescaled.beta+1)**2)
        grad = grad_input/(SuperSpike_rescaled.beta *
                           torch.abs(input)+1.0)**2 / rescale_val
        return grad


class MultiSpike(torch.autograd.Function):
    """
    Autograd MultiSpike nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta (default=100).
    """
    beta = 100.0
    maxspk = 10.0

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations. 
        """
        ctx.save_for_backward(input)
        out = nn.functional.hardtanh(
            torch.round(input+0.5), 0.0, MultiSpike.maxspk)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(MultiSpike.beta*torch.abs(input -
                           torch.relu(torch.round(input)))+1.0)**2
        return grad


class SuperSpike_asymptote(torch.autograd.Function):
    """
    Autograd SuperSpike nonlinearity implementation with asymptotic behavior of step.

    The steepness parameter beta can be accessed via the static member
    self.beta (default=100).
    """
    beta = 100.0

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations. 
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = SuperSpike_asymptote.beta*grad_input / \
            (SuperSpike_asymptote.beta*torch.abs(input)+1.0)**2
        return grad


class TanhSpike(torch.autograd.Function):
    """
    Autograd Tanh et al. nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta (default=100).
    """
    beta = 100.0

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations. 
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        beta = TanhSpike.beta
        grad = grad_input*(1.0+(1.0-torch.tanh(input*beta)**2))
        return grad


class SigmoidSpike(torch.autograd.Function):
    """
    Autograd surrogate gradient nonlinearity implementation which uses the derivative of a sigmoid in the backward pass.

    The steepness parameter beta can be accessed via the static member self.beta (default=100).
    """
    beta = 2

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations. 
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        sig = torch.sigmoid(SigmoidSpike.beta*input)
        dsig = sig*(1.0-sig)
        grad = grad_input*dsig
        return grad


class StochasticSpike(torch.autograd.Function):
    """
    Stochastic spike implementation, where the probability of a spike follows a sigmoid. The backward path uses a straight-through estimator, where the derivative of a hard threshold is 1 and the derivative of the probability of spiking (derivative of a sigmoid) is consiedered in the backward pass.

    The steepness parameter beta can be accessed via the static member self.beta (default=20).
    """
    beta = 2

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations. 
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        prob = torch.sigmoid(StochasticSpike.beta*input)
        if prob.get_device() < 0:
            p = torch.rand(size=prob.shape)
        else:
            p = torch.rand(size=prob.shape, device=prob.get_device())
        out[prob > p] = 1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        sig = torch.sigmoid(StochasticSpike.beta*input)
        dsig = sig*(1.0-sig)
        grad = grad_input*dsig
        return grad


class ExponentialStochasticSpike(torch.autograd.Function):
    """
    Stochastic spike implementation, where the probability of a spike follows an exponential. The backward pass uses the derivative of the probability of spiking.

    The parameters p0 and delta_u can be accessed via the static member self.p0 (default=0.01) or self.delta_u (default=0.2).
    """
    p0 = 0.01
    delta_u = 0.013
    eps_0 = 0.267

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations. 
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        prob = ExponentialStochasticSpike.p0 * \
            torch.exp(input*ExponentialStochasticSpike.eps_0 /
                      ExponentialStochasticSpike.delta_u)
        if prob.get_device() < 0:
            p = torch.rand(size=prob.shape)
        else:
            p = torch.rand(size=prob.shape, device=prob.get_device())
        out[prob > p] = 1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        p = ExponentialStochasticSpike.p0 * \
            torch.exp(input*ExponentialStochasticSpike.eps_0 /
                      ExponentialStochasticSpike.delta_u)
        dp = p / ExponentialStochasticSpike.delta_u*ExponentialStochasticSpike.eps_0
        grad = grad_input*dp
        return grad


class MultilayerSpikerSpike(torch.autograd.Function):
    """
    Stochastic spike implementation, where the probability of a spike follows an exponential. The backward pass uses the spiketrain itself as the derivative of the spike train.

    The parameters p0 and delta_u can be accessed via the static member self.p0 (default=0.01) or self.delta_u (default=0.2).
    """
    p0 = 0.01
    delta_u = 0.013
    eps_0 = 0.267

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations. 
        """
        out = torch.zeros_like(input)
        prob = MultilayerSpikerSpike.p0 * \
            torch.exp(input*MultilayerSpikerSpike.eps_0 /
                      MultilayerSpikerSpike.delta_u)
        if prob.get_device() < 0:
            p = torch.rand(size=prob.shape)
        else:
            p = torch.rand(size=prob.shape, device=prob.get_device())
        out[prob > p] = 1
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        out, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input*out
        return grad


class SigmoidalMultilayerSpikerSpike(torch.autograd.Function):
    """
    Stochastic spike implementation, where the probability of a spike follows a sigmoid. The backward pass uses the spiketrain itself as the derivative of the spike train.

    The steepness parameter beta can be accessed via the static member self.beta (default=20).
    """
    beta = 2

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations. 
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        prob = torch.sigmoid(SigmoidalMultilayerSpikerSpike.beta*input)
        if prob.get_device() < 0:
            p = torch.rand(size=prob.shape)
        else:
            p = torch.rand(size=prob.shape, device=prob.get_device())
        out[prob > p] = 1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        out, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input*out
        return grad


class EsserSpike(torch.autograd.Function):
    """
    Autograd surrogate gradient nonlinearity implementation which uses piecewise linear pseudo derivative in the backward pass as suggested in:

        Esser, S.K., Merolla, P.A., Arthur, J.V., Cassidy, A.S., Appuswamy, R.,
        Andreopoulos, A., Berg, D.J., McKinstry, J.L., Melano, T., Barch, D.R.,
        et al. (2016). Convolutional networks for fast, energy-efficient
        neuromorphic computing. Proc Natl Acad Sci U S A 113, 11441–11446.
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5068316/

    The steepness parameter beta can be accessed via the static member self.beta (default=1.0).
    """
    beta = 1.0

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations. 
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * \
            torch.max(torch.zeros_like(input), 1.0 -
                      torch.abs(EsserSpike.beta*input))
        return grad


class HardTanhSpike(torch.autograd.Function):
    """
    Autograd Esser et al. nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta (default=100).
    """
    beta = 100.0

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations. 
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        beta = HardTanhSpike.beta
        grad = grad_input*(1.0+torch.nn.functional.hardtanh(input*beta))
        return grad


class SuperSpike_norm(torch.autograd.Function):
    """
    Autograd SuperSpike nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta (default=100).
    """
    beta = 100.0
    xi = 1e-2

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations. 
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SuperSpike_norm.beta*torch.abs(input)+1.0)**2
        # standardize gradient
        standard_grad = grad/(SuperSpike_norm.xi +
                              torch.norm(torch.mean(grad, dim=0)))
        return standard_grad
