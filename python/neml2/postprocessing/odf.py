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
        kernel (Kernel): kernel function
    """

    def __init__(self, X, kernel):
        super().__init__(X)
        self.kernel = kernel

    def forward(self, Y):
        """Calculate the probability density at each point in Y

        Args:
            Y (neml2.tensors.Rot): rotations with arbitrary batch shape

        Returns:
            torch.tensor with the probabilities
        """
        dist = self.X.dist(Y.batch.unsqueeze(-1)).torch()

        import matplotlib.pyplot as plt

        return torch.mean(
            self.kernel(torch.cos(dist / 2.0)),
            dim=-1,
        )


class Kernel(torch.nn.Module):
    """Parent class for kernels for KDE reconstruction

    Args:
        h (torch.tensor): half-width
    """

    def __init__(self, h):
        super().__init__()
        self.h = h


class DeLaValleePoussinKernel(Kernel):
    """De La Vallee Poussin kernel, according to MTEX"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kappa = 0.5 * math.log(0.5) / torch.log(torch.cos(self.h / 2))
        self.c = beta(
            torch.tensor(1.5, device=self.h.device),
            torch.tensor(0.5, device=self.h.device),
        ) / beta(torch.tensor(1.5, device=self.h.device), self.kappa + 0.5)

    def forward(self, X):
        """Evaluate the kernel

        Args:
            X (torch.tensor):
        """
        return self.c * X ** (2 * self.kappa)


def beta(z1, z2):
    """Calculate the beta function

    Args:
        z1 (torch.tensor): first input
        z2 (torch.tensor): second input
    """
    return gamma(z1) * gamma(z2) / gamma(z1 + z2)


def gamma(z):
    """Calculate the gamma function

    Args:
        z (torch.tensor): input
    """
    return torch.exp(torch.special.gammaln(z))
