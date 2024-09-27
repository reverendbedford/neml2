import torch

from neml2 import crystallography
from neml2 import tensors


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
