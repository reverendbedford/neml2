from math import sqrt

import torch

import matplotlib.pyplot as plt

from neml2 import crystallography
from neml2 import tensors


class StereographicProjection:
    """Stereographic projection"""

    def __init__(self):
        self.limit = 1.0

    def __call__(self, v):
        """Project from 3d points on sphere to 2d

        Args:
            v (torch.tensor): tensor of shape (..., 3)
        """
        return torch.stack(
            [v[..., 0] / (1.0 + v[..., 2]), v[..., 1] / (1.0 + v[..., 2])], dim=-1
        )


class LambertProjection:
    """Lambert equal area projection"""

    def __init__(self):
        self.limit = sqrt(2.0)

    def __call__(self, v):
        """Project from 3d points on sphere to 2d

        Args:
            v (torch.tensor): tensor of shape (..., 3)
        """
        return torch.stack(
            [
                torch.sqrt(2.0 / (1.0 + v[..., 2])) * v[..., 0],
                torch.sqrt(2.0 / (1.0 + v[..., 2])) * v[..., 1],
            ],
            dim=-1,
        )


available_projections = {
    "stereographic": StereographicProjection(),
    "lambert": LambertProjection(),
}


def symmetry_operators_as_R2(orbifold, device=torch.device("cpu")):
    """Return the symmetry operators for a given symmetry group as a batch of rank two tensors

    Args:
        orbifold (str): symmetry group in orbifold notation

    Keyword Args:
        device (torch.device): which device to place the tensors
    """
    return crystallography.symmetry_operations_from_orbifold(orbifold, device=device)


def pole_figure_points(
    orientations,
    pole,
    projection="stereographic",
    crystal_symmetry="1",
    sample_symmetry="1",
):
    """Project crystal orientations to points on a pole figure

    Args:
        orientations (torch.tensor or neml2.tensors.Rot): tensor of orientations as *modified* Rodrigues
            parameters in the *active* convention with arbitrary batch shape.
        pole (torch.tensor or neml2.tensors.Vec): pole to project, must broadcast with orientations

    Keyword Args:
        projection (str): which polar projection to use, options are "stereographic" and "lambert"
        crystal_symmetry (str): string giving the orbifold notation for the crystal symmetry to apply to the base orientations, default "1"
        sample_symmetry (str): string giving the orbifold notation for the sample symmetry to apply to the projected points, default "1"

    Returns:
        torch.tensor: tensor with same batch shape as orientations but with end dimension (2,) giving each point
    """
    # Do some setup
    if not isinstance(pole, tensors.Vec):
        pole = tensors.Vec(pole)
    pole = pole / pole.norm()
    if not isinstance(orientations, tensors.Rot):
        orientations = tensors.Rot(orientations)

    # Get all the equivalent poles
    crystal_symmetry_operators = symmetry_operators_as_R2(crystal_symmetry)
    equivalent_poles = crystal_symmetry_operators * pole

    # Move from crystal to sample
    sample_poles = equivalent_poles.rotate(orientations.batch.unsqueeze(-1))
    # Apply sample symmetry
    sample_symmetry_operators = symmetry_operators_as_R2(sample_symmetry)
    sample_poles = sample_symmetry_operators * sample_poles.batch.unsqueeze(-1)

    # For my reference, at this point we have a tensor of (arbitrary_batch_shape,) + (crystal_symmetry,) + (sample_symmetry,) + (3,)
    sample_poles = sample_poles.torch()

    # Eliminate poles on the lower hemisphere
    sample_poles = sample_poles[sample_poles[..., 2] >= 0.0]

    # Project and return
    projection = available_projections[projection]
    return projection(sample_poles)


def cart2polar(v):
    """Convert cartesian 2D points to polar coordinates

    Args:
        v (torch.tensor): tensor of shape (...,2)
    """
    return torch.stack(
        [torch.atan2(v[..., 1], v[..., 0]), torch.norm(v, dim=-1)], dim=-1
    )


def pretty_plot_polefigure(
    orientations,
    pole,
    projection="stereographic",
    crystal_symmetry="1",
    sample_symmetry="1",
    point_size=10.0,
):
    """Project and then make a pretty plot for a polefigure

    Args:
        orientations (torch.tensor or neml2.tensors.Rot): tensor of orientations as *modified* Rodrigues
            parameters in the *active* convention with arbitrary batch shape.
        pole (torch.tensor or neml2.tensors.Vec): pole to project, must broadcast with orientations

    Keyword Args:
        projection (str): which polar projection to use, options are "stereographic" and "lambert"
        crystal_symmetry (str): string giving the orbifold notation for the crystal symmetry to apply to the base orientations, default "1"
        sample_symmetry (str): string giving the orbifold notation for the sample symmetry to apply to the projected points, default "1"
        point_size (float): size of matplotlib points to plot
    """
    points = pole_figure_points(
        orientations, pole, projection, crystal_symmetry, sample_symmetry
    )
    projection = available_projections[projection]

    polar = cart2polar(points)

    # Plot
    ax = plt.subplot(111, projection="polar")
    ax.scatter(polar[..., 0].cpu(), polar[..., 1].cpu(), c="k", s=point_size)

    # Make the graph nice
    plt.ylim([0, projection.limit])
    ax.grid(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.xaxis.set_minor_locator(plt.NullLocator())
