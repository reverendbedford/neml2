#pragma once

#include "models/Model.h"

/// Drive a model in strain control
class StructuralStrainControlDriver {
 public:
  // TODO: Add temperature as an input
  StructuralStrainControlDriver(const neml2::Model & model,
                                torch::Tensor time,
                                torch::Tensor strain);

  /// Actually run and return the results
  virtual std::tuple<std::vector<neml2::LabeledVector>, std::vector<neml2::LabeledVector>> run();

protected:
  virtual neml2::LabeledVector solve_step(neml2::LabeledVector in,
                                          neml2::TorchSize i,
                                          std::vector<neml2::LabeledVector> & all_inputs,
                                          std::vector<neml2::LabeledVector> & all_outputs) const;

  const neml2::Model & _model;
  torch::Tensor _time;
  torch::Tensor _strain;
  neml2::TorchSize _nsteps;
  neml2::TorchSize _nbatch;

};
