#pragma once

#include "models/Model.h"

/// Drive a model under uniaxial strain conditions
class UniaxialStrainStructuralDriver : public LabeledAxisInterface
{
public:
  // TODO: add temperature here and elsewhere
  UniaxialStrainStructuralDriver(Model & model,
                                 Scalar max_strain,
                                 Scalar end_time,
                                 TorchSize nsteps);

  /// Actually run the model and store the results
  std::tuple<std::vector<LabeledVector>, std::vector<LabeledVector>> run();

protected:
  LabeledVector solve_step(LabeledVector in,
                           TorchSize i,
                           std::vector<LabeledVector> & all_inputs,
                           std::vector<LabeledVector> & all_outputs) const;

  Model & _model;
  Scalar _max_strain;
  Scalar _end_time;
  TorchSize _nsteps;
  TorchSize _nbatch;
};
