import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class LossStack:
    def __init__(self):
        pass

    def log_py_given_x(self, output):
        raise NotImplemented()

    def get_metric_names(self):
        raise NotImplemented()

    def compute_loss(self, output, targets):
        raise NotImplemented()

    def predict(self, output):
        raise NotImplemented()


class MaxOverTimeCrossEntropy(LossStack):
    """Readout stack that employs the max-over-time reduction strategy paired with categorical cross entropy."""

    def __init__(self, time_dimension=1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.neg_log_likelihood_loss = nn.NLLLoss()
        self.time_dim = time_dimension

    def acc_fn(self, log_p_y, target_labels):
        """Computes classification accuracy from log_p_y and corresponding target labels

        Args:
            log_p_y: The log softmax output (log p_y_given_x) of the model.
            target_labels: The integer target labels (not one hot encoding).

        Returns:
            Float of mean classification accuracy.
        """
        _, pred_labels = torch.max(log_p_y, dim=self.time_dim)
        a = pred_labels == target_labels
        return (1.0 * a.cpu().numpy()).mean()

    def get_metric_names(self):
        return ["acc"]

    def compute_loss(self, output, targets):
        """Computes crossentropy loss on softmax defined over maxpooling over time"""
        ma, _ = torch.max(output, self.time_dim)  # reduce along time with max
        log_p_y = self.log_softmax(ma)
        loss_value = self.neg_log_likelihood_loss(
            log_p_y, targets
        )  # compute supervised loss
        acc_val = self.acc_fn(log_p_y, targets)
        self.metrics = [acc_val.item()]
        return loss_value

    def log_py_given_x(self, output):
        ma, _ = torch.max(output, self.time_dim)  # reduce along time with max
        log_p_y = self.log_softmax(ma)
        return log_p_y

    def predict(self, output):
        _, pred_labels = torch.max(self.log_py_given_x(output), dim=1)
        return pred_labels

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


class MaxOverTimeFocalLoss(LossStack):
    """Readout stack that employs the max-over-time reduction strategy paired with focal loss."""

    def __init__(
        self, gamma=0.0, eps=1e-7, samples_per_class=None, beta=0.99, time_dimension=1
    ):
        super().__init__()
        self.time_dim = time_dimension
        self.eps = eps
        self.gamma = gamma
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.weights = None
        if samples_per_class is not None:
            eff_num = 1.0 - np.power(beta, samples_per_class)
            weights = (1.0 - beta) / np.array(eff_num)
            weights = weights / np.sum(weights) * len(samples_per_class)
            print("weights: ", weights)
            self.weights = torch.tensor(weights).float()

    def acc_fn(self, log_p_y, target_labels):
        """Computes classification accuracy from log_p_y and corresponding target labels

        Args:
            log_p_y: The log softmax output (log p_y_given_x) of the model.
            target_labels: The integer target labels (not one hot encoding).

        Returns:
            Float of mean classification accuracy.
        """
        _, pred_labels = torch.max(log_p_y, dim=self.time_dim)
        a = pred_labels == target_labels
        return (1.0 * a.cpu().numpy()).mean()

    def get_metric_names(self):
        return ["acc"]

    def compute_loss(self, output, targets):
        # reduce along time with max
        ma, _ = torch.max(output, dim=self.time_dim)
        y = F.one_hot(targets, ma.size(-1))

        logit = F.softmax(ma, dim=-1)
        logit = logit.clamp(self.eps, 1.0 - self.eps)

        loss = -1.0 * y * torch.log(logit)  # cross entropy loss
        loss = loss * (1.0 - logit) ** self.gamma  # change to focal loss

        if self.weights is not None:
            w = self.weights.to(loss.device)
            w = w.unsqueeze(0)
            w = w.repeat(y.shape[0], 1) * y
            w = w.sum(1)
            w = w.unsqueeze(1)
            w = w.repeat(1, ma.size(-1))
            loss = w * loss

        acc_val = self.acc_fn(logit, targets)
        self.metrics = [acc_val.item()]
        return torch.mean(loss)

    def log_py_given_x(self, output):
        ma, _ = torch.max(output, self.time_dim)  # reduce along time with max
        log_p_y = self.log_softmax(ma)
        return log_p_y

    def predict(self, output):
        _, pred_labels = torch.max(self.log_py_given_x(output), dim=1)
        return pred_labels

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


class SumOverTimeCrossEntropy(LossStack):
    """Loss stack that employs the sum-over-time reduction strategy paired with categorical cross entropy."""

    def __init__(self, time_dimension=1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.neg_log_likelihood_loss = nn.NLLLoss()
        self.time_dim = time_dimension

    def acc_fn(self, log_p_y, target_labels):
        """Computes classification accuracy from log_p_y and corresponding target labels

        Args:
            log_p_y: The log softmax output (log p_y_given_x) of the model.
            target_labels: The integer target labels (not one hot encoding).

        Returns:
            Float of mean classification accuracy.
        """
        _, pred_labels = torch.max(log_p_y, dim=self.time_dim)
        a = pred_labels == target_labels
        return (1.0 * a.cpu().numpy()).mean()

    def get_metric_names(self):
        return ["acc"]

    def compute_loss(self, output, targets):
        """Computes crossentropy loss on softmax defined over sum over time"""
        su = torch.sum(output, self.time_dim)  # reduce along time with sum
        log_p_y = self.log_softmax(su)
        loss_value = self.neg_log_likelihood_loss(
            log_p_y, targets
        )  # compute supervised loss
        acc_val = self.acc_fn(log_p_y, targets)
        self.metrics = [acc_val.item()]
        return loss_value

    def log_py_given_x(self, output):
        su = torch.sum(output, self.time_dim)  # reduce along time with sum
        log_p_y = self.log_softmax(su)
        return log_p_y

    def predict(self, output):
        _, pred_labels = torch.max(self.log_py_given_x(output), dim=1)
        return pred_labels

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


class LastStepCrossEntropy(LossStack):
    """Computes crossentropy loss on last time frame of the network"""

    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.neg_log_likelihood_loss = nn.NLLLoss()

    def acc_fn(self, log_p_y, target_labels):
        """Computes classification accuracy from log_p_y and corresponding target labels

        Args:
            log_p_y: The log softmax output (log p_y_given_x) of the model.
            target_labels: The integer target labels (not one hot encoding).

        Returns:
            Float of mean classification accuracy.
        """
        _, pred_labels = torch.max(log_p_y, dim=1)
        a = pred_labels == target_labels
        return a.cpu().numpy().mean()

    def get_metric_names(self):
        return ["acc"]

    def compute_loss(self, output, targets):
        """Computes crossentropy loss on softmax defined over maxpooling over time"""
        log_p_y = self.log_softmax(output[:, -1])
        loss_value = self.neg_log_likelihood_loss(
            log_p_y, targets
        )  # compute supervised loss
        acc_val = self.acc_fn(log_p_y, targets)
        self.metrics = [acc_val.item()]
        return loss_value

    def log_py_given_x(self, output):
        log_p_y = self.log_softmax(output[:, -1])
        return log_p_y

    def predict(self, output):
        _, pred_labels = torch.max(self.log_py_given_x(output), dim=1)
        return pred_labels

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


class EveryStepCrossEntropy(LossStack):
    """Computes crossentropy loss on every time frame of the network"""

    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.neg_log_likelihood_loss = nn.NLLLoss()

    def acc_fn(self, log_p_y, target_labels):
        """Computes classification accuracy from log_p_y and corresponding target labels

        Args:
            log_p_y: The log softmax output (log p_y_given_x) of the model.
            target_labels: The integer target labels (not one hot encoding).

        Returns:
            Float of mean classification accuracy.
        """
        _, pred_labels = torch.max(log_p_y, dim=1)
        a = pred_labels == target_labels
        return a.cpu().numpy().mean()

    def get_metric_names(self):
        return ["acc"]

    def compute_loss(self, output, targets):
        """Computes crossentropy loss on softmax defined over maxpooling over time"""
        log_p_y = self.log_softmax(output)
        loss_value = self.neg_log_likelihood_loss(
            log_p_y, targets
        )  # compute supervised loss
        acc_val = self.acc_fn(log_p_y, targets)
        self.metrics = [acc_val.item()]
        return loss_value

    def log_py_given_x(self, output):
        log_p_y = self.log_softmax(output)
        return log_p_y

    def predict(self, output):
        _, pred_labels = torch.max(self.log_py_given_x(output), dim=1)
        return pred_labels

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


class MeanSquareError(LossStack):
    def __init__(self, mask=None):
        """
        Args:

            mask: A ``don't-care'' mask which can be aplied to part of the output
        """
        super().__init__()
        self.msqe_loss = nn.MSELoss()
        self.mask = mask

    def get_metric_names(self):
        return []

    def compute_loss(self, output, target):
        """Computes MSQE loss between output and target."""
        if self.mask is None:
            loss_value = self.msqe_loss(output, target)
        else:
            loss_value = self.msqe_loss(
                output * self.mask.expand_as(output),
                target * self.mask.expand_as(output),
            )
        self.metrics = []
        return loss_value

    def predict(self, output):
        return output  # here we just return the network output

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


class DictMeanSquareError(MeanSquareError):
    """Like MeanSquareError, but uses a dictionary of possible output
    patterns which can be kept in the GPU memory."""

    def __init__(self, target_patterns, mask=None):
        super().__init__(mask)
        self.dict_ = target_patterns

    def compute_loss(self, output, targets):
        """Computes MSQE loss between output and target."""
        local_targets = [self.dict_[idx] for idx in targets]
        return super().compute_loss(output, torch.stack(local_targets, dim=0))

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


class DoubleData_MaxOverTimeCrossEntropy(MaxOverTimeCrossEntropy):

    """Readout stack that employs the max-over-time reduction strategy paired with categorical cross entropy."""

    def __init__(self, time_dimension=1, frac=0.5):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.neg_log_likelihood_loss = nn.NLLLoss()
        self.time_dim = time_dimension
        self.frac = frac

    def acc_fn(self, log_p_y, target_labels):
        """Computes classification accuracy from log_p_y and corresponding target labels

        Args:
            log_p_y: The log softmax output (log p_y_given_x) of the model.
            target_labels: The integer target labels (not one hot encoding).

        Returns:
            Float of mean classification accuracy.
        """
        _, pred_labels0 = torch.max(log_p_y[0], dim=self.time_dim)
        _, pred_labels1 = torch.max(log_p_y[1], dim=self.time_dim)

        a = self.frac * (pred_labels0 == target_labels[:, 0]) + (1 - self.frac) * (
            pred_labels1 == target_labels[:, 1]
        )
        return (1.0 * a.cpu().numpy()).mean()

    def get_metric_names(self):
        return ["acc"]

    def compute_loss(self, output, targets):
        """Computes crossentropy loss on softmax defined over maxpooling over time"""
        ma0, _ = torch.max(output[0], self.time_dim)  # reduce along time with max
        ma1, _ = torch.max(output[1], self.time_dim)  # reduce along time with max
        log_p_y0 = self.log_softmax(ma0)
        log_p_y1 = self.log_softmax(ma1)
        loss_value0 = self.neg_log_likelihood_loss(
            log_p_y0, targets[:, 0]
        )  # compute supervised loss
        loss_value1 = self.neg_log_likelihood_loss(log_p_y1, targets[:, 1])
        acc_val = self.acc_fn([log_p_y0, log_p_y1], targets)
        self.metrics = [acc_val.item()]
        return loss_value0 + loss_value1

    def log_py_given_x(self, output):
        ma, _ = torch.max(output, self.time_dim)  # reduce along time with max
        log_p_y = self.log_softmax(ma)
        return log_p_y

    def predict(self, output):
        _, pred_labels0 = torch.max(self.log_py_given_x(output[0]), dim=1)
        _, pred_labels1 = torch.max(self.log_py_given_x(output[1]), dim=1)
        return [pred_labels0, pred_labels1]

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


# For backward compatibility
TemporalCrossEntropyReadoutStack = MaxOverTimeCrossEntropy
