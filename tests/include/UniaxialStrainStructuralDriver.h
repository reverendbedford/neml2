#pragma once

#include "StructuralStrainControlDriver.h"

/// Drive a model under uniaxial strain conditions
class UniaxialStrainStructuralDriver : public StructuralStrainControlDriver
{
public:
  // TODO: add temperature here and elsewhere
  UniaxialStrainStructuralDriver(const neml2::Model & model,
                                 neml2::Scalar max_strain,
                                 neml2::Scalar end_time,
                                 neml2::TorchSize nsteps);

protected:
  neml2::Scalar _max_strain;
  neml2::Scalar _end_time;
};

/// A batched version of torch::linspace
torch::Tensor batched_linspace(torch::Tensor start, torch::Tensor stop, neml2::TorchSize nsteps);
