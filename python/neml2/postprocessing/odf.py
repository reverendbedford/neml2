import torch
import math


class ODF(torch.nn.Module):
    """Parent class for Orientation Distribution Functions

    Args:
        X (neml2.tensors.Rot): rotations, must have a single batch dimension
    """

    def __init__(self, X):
        super().__init__()
        assert X.batch.dim() == 1
        self.X = X
        self.n = X.batch.shape[0]


class KDEODF(ODF):
    """ODF represented from a Kernel Density Estimate

    Args:
        X (neml2.tensors.Rot): rotations, must have a single batch dimension
        h (torch.tensor): bandwidth
        kernel (Kernel): kernel function
    """

    def __init__(self, X, h, kernel):
        super().__init__(X)
        self.h = h
        self.kernel = kernel

    def forward(self, Y):
        """Calculate the probability density at each point in Y

        Args:
            Y (neml2.tensors.Rot): rotations with arbitrary batch shape

        Returns:
            torch.tensor with the probabilities
        """
        return torch.sum(
            self.kernel(self.X.dist(Y.batch.unsqueeze(-1)).torch() / self.h), dim=-1
        ) / (self.n * self.h)


class Kernel(torch.nn.Module):
    """Parent class for kernels for KDE reconstruction"""

    def __init__(self):
        super().__init__()


class CosineKernel(Kernel):
    """Cosine kernel"""

    def forward(self, X):
        """Evaluate the kernel

        Args:
            X (torch.tensor):
        """
        return torch.cos(X / 2.0) / 2.0
