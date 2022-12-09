#pragma once

#include "models/Model.h"

/// Drive a model under uniaxial strain conditions
class UniaxialStrainStructuralDriver
{
public:
  // TODO: add temperature here and elsewhere
  UniaxialStrainStructuralDriver(const neml2::Model & model,
                                 neml2::Scalar max_strain,
                                 neml2::Scalar end_time,
                                 neml2::TorchSize nsteps);

  /// Actually run the model and store the results
  std::tuple<std::vector<neml2::LabeledVector>, std::vector<neml2::LabeledVector>> run();

protected:
  neml2::LabeledVector solve_step(neml2::LabeledVector in,
                                  neml2::TorchSize i,
                                  std::vector<neml2::LabeledVector> & all_inputs,
                                  std::vector<neml2::LabeledVector> & all_outputs) const;

  const neml2::Model & _model;
  neml2::Scalar _max_strain;
  neml2::Scalar _end_time;
  neml2::TorchSize _nsteps;
  neml2::TorchSize _nbatch;
};
