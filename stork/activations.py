import torch
import torch.nn as nn


class SuperSpike(torch.autograd.Function):
    """
    Autograd SuperSpike nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta.
    """

    beta = 20

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1
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
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SuperSpike.beta * torch.abs(input) + 1) ** 2
        return grad


class CustomSpike(torch.autograd.Function):
    """
    Customizable autograd SuperSpike nonlinearity implementation that allows for escape noise in the
    forward path and uses a surrogate gradient on the backward path.

    If escape_noise_type is "step", forward will be a step function and otherwise sampled given the
    indicated escape noise function.

    Supported surrogate types ["SuperSpike", "sigmoid", "MultilayerSpiker", "exponential"]
    Supported escape noise types are ["step", "sigmoid", "exponential"]

    both parameters are dicts, that may contain the following parameters ["beta", "p0", "delta_u"]
    """

    escape_noise_type = "step"
    escape_noise_params = {"beta": 10, "p0": 0.01, "delta_u": 0.133}
    surrogate_type = "SuperSpike"
    surrogate_params = {"beta": 10, "p0": 0.01, "delta_u": 0.133}

    @staticmethod
    def forward(ctx, input):
        if CustomSpike.escape_noise_type == "step":
            return CustomSpike.forward_step(ctx, input)
        elif CustomSpike.escape_noise_type == "sigmoid":
            return CustomSpike.forward_sigmoid_s(ctx, input)
        elif CustomSpike.escape_noise_type == "exponential":
            return CustomSpike.forward_exponential_s(ctx, input)
        elif CustomSpike.escape_noise_type == "SuperSpike":
            return CustomSpike.forward_superspike_s(ctx, input)
        else:
            raise ValueError(
                "Escape noise type not supported. Please chose one of the following: step, sigmoid, exponential"
            )

    @staticmethod
    def backward(ctx, grad_output):
        if CustomSpike.surrogate_type == "SuperSpike":
            return CustomSpike.backward_superspike(ctx, grad_output)
        elif CustomSpike.surrogate_type == "sigmoid":
            return CustomSpike.backward_sigmoid(ctx, grad_output)
        elif CustomSpike.surrogate_type == "scaled_sigmoid":
            return CustomSpike.backward_scaled_sigmoid(ctx, grad_output)
        elif CustomSpike.surrogate_type == "MultilayerSpiker":
            return CustomSpike.backward_multilayerspiker(ctx, grad_output)
        elif CustomSpike.surrogate_type == "exponential":
            return CustomSpike.backward_exponential(ctx, grad_output)
        else:
            raise ValueError(
                "Surrogate type not supported. Please chose one of the following: SuperSpike, sigmoid, MultilayerSpiker, exponential"
            )

    @staticmethod
    def forward_step(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1
        return out

    @staticmethod
    def forward_sigmoid_s(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output using a sigmoidal probability of spiking.
        ctx is the context object that is used to stash information for backward
        pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        prob = torch.sigmoid(CustomSpike.escape_noise_params["beta"] * input)
        if prob.get_device() < 0:
            p = torch.rand(size=prob.shape)
        else:
            p = torch.rand(size=prob.shape, device=prob.get_device())
        out[prob > p] = 1
        return out

    @staticmethod
    def forward_exponential_s(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output using an exponential probability of spiking.
        ctx is the context object that is used to stash information for backward
        pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        prob = CustomSpike.escape_noise_params["p0"] * torch.exp(
            input / CustomSpike.escape_noise_params["delta_u"]
        )
        if prob.get_device() < 0:
            p = torch.rand(size=prob.shape)
        else:
            p = torch.rand(size=prob.shape, device=prob.get_device())
        out[prob > p] = 1
        return out

    @staticmethod
    def forward_superspike_s(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output using a fast sigmoid probability scaled by 1/beta of spiking.
        ctx is the context object that is used to stash information for backward
        pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        x = CustomSpike.escape_noise_params["beta"] * input
        prob = x / (1 + torch.abs(x)) / CustomSpike.escape_noise_params["beta"]
        if prob.get_device() < 0:
            p = torch.rand(size=prob.shape)
        else:
            p = torch.rand(size=prob.shape, device=prob.get_device())
        out[prob > p] = 1
        return out

    @staticmethod
    def backward_superspike(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input
            / (CustomSpike.surrogate_params["beta"] * torch.abs(input) + 1) ** 2
        )
        return grad

    @staticmethod
    def backward_sigmoid(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input, considering a the gradient of a
        sigmoid function as the surrogate gradient.
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        sig = torch.sigmoid(CustomSpike.surrogate_params["beta"] * input)
        dsig = CustomSpike.surrogate_params["beta"] * sig * (1 - sig)
        grad = grad_input * dsig
        return grad

    @staticmethod
    def backward_scaled_sigmoid(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input, considering a the gradient of a
        sigmoid function as the surrogate gradient.
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        sig = torch.sigmoid(CustomSpike.surrogate_params["beta"] * input)
        dsig = sig * (1 - sig)
        grad = grad_input * dsig
        return grad

    @staticmethod
    def backward_multilayerspiker(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we replace the derivative of a spiketrain
        by the spiketrain itself (see Gardner et al., 2015)
        """
        (out,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * out
        return grad

    @staticmethod
    def backward_exponential(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the gradient using a
        straight-through estimator, meaning the derivative of the hard threshold
        is replaced by one, while only using the derivative of the probability of
        spiking.
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        p = (
            CustomSpike.surrogate_params["p0"]
            / CustomSpike.surrogate_params["delta_u"]
            * torch.exp(input / CustomSpike.surrogate_params["delta_u"])
        )
        grad = grad_input * p
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
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input
            / (SuperSpike_MemClamp.beta * torch.abs(torch.relu(-input)) + 1.0) ** 2
        )
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
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        rescale_val = 1 / ((SuperSpike_rescaled.beta + 1) ** 2)
        grad = (
            grad_input
            / (SuperSpike_rescaled.beta * torch.abs(input) + 1.0) ** 2
            / rescale_val
        )
        return grad


class MultiSpike(torch.autograd.Function):
    """
    Autograd MultiSpike nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta (default=100).
    """

    beta = 100
    maxspk = 10

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = nn.functional.hardtanh(torch.round(input + 0.5), 0, MultiSpike.maxspk)
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
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input
            / (MultiSpike.beta * torch.abs(input - torch.relu(torch.round(input))) + 1)
            ** 2
        )
        return grad


class SuperSpike_asymptote(torch.autograd.Function):
    """
    Autograd SuperSpike nonlinearity implementation with asymptotic behavior of step.

    The steepness parameter beta can be accessed via the static member
    self.beta (default=100).
    """

    beta = 100

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1
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
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            SuperSpike_asymptote.beta
            * grad_input
            / (SuperSpike_asymptote.beta * torch.abs(input) + 1) ** 2
        )
        return grad


class TanhSpike(torch.autograd.Function):
    """
    Autograd Tanh et al. nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta (default=100).
    """

    beta = 100

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1
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
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        beta = TanhSpike.beta
        grad = grad_input * (1 + (1 - torch.tanh(input * beta) ** 2))
        return grad


class EsserSpike(torch.autograd.Function):
    """
    Autograd surrogate gradient nonlinearity implementation which uses piecewise linear pseudo derivative in the backward pass as suggested in:

        Esser, S.K., Merolla, P.A., Arthur, J.V., Cassidy, A.S., Appuswamy, R.,
        Andreopoulos, A., Berg, D.J., McKinstry, J.L., Melano, T., Barch, D.R.,
        et al. (2016). Convolutional networks for fast, energy-efficient
        neuromorphic computing. Proc Natl Acad Sci U S A 113, 11441–11446.
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5068316/

    The steepness parameter beta can be accessed via the static member self.beta (default=1).
    """

    beta = 1

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1
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
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * torch.max(
            torch.zeros_like(input), 1 - torch.abs(EsserSpike.beta * input)
        )
        return grad


class HardTanhSpike(torch.autograd.Function):
    """
    Autograd Esser et al. nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta (default=100).
    """

    beta = 100

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1
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
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        beta = HardTanhSpike.beta
        grad = grad_input * (1 + torch.nn.functional.hardtanh(input * beta))
        return grad


class SuperSpike_norm(torch.autograd.Function):
    """
    Autograd SuperSpike nonlinearity implementation.

    The steepness parameter beta can be accessed via the static member
    self.beta (default=100).
    """

    beta = 100
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
        out[input > 0] = 1
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
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SuperSpike_norm.beta * torch.abs(input) + 1) ** 2
        # standardize gradient
        standard_grad = grad / (
            SuperSpike_norm.xi + torch.norm(torch.mean(grad, dim=0))
        )
        return standard_grad
