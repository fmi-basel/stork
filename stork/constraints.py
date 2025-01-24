import torch


class WeightConstraint:
    """Abstract base class for weight constraints."""

    def __init__(self):
        pass

    def apply(self):
        pass


class MinMaxConstraint(WeightConstraint):
    def __init__(self, min=None, max=None):
        """Implements a min max constraint for connection weights.

        Args:
            connection: The connection object to apply the constraint to
            min: The minimum weight
            max: The maximum weight
        """
        super().__init__()
        self.min = min
        self.max = max

    def apply(self, weight):
        with torch.no_grad():
            torch.clamp_(weight, min=self.min, max=self.max)
