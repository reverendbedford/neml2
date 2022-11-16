#pragma once

#include "ConstitutiveModel.h"

/// Drive a model under uniaxial strain conditions
class UniaxialStrainStructuralDriver
{
public:
  // TODO: add temperature here and elsewhere
  UniaxialStrainStructuralDriver(ConstitutiveModel & model,
                                 torch::Tensor max_strain,
                                 torch::Tensor strain_rate);

  /// Actually run the model and store the results
  void run(TorchSize nsteps);

  /// Getter for the model forces
  const torch::Tensor & forces() const { return _forces; };

  /// Getter for the model state
  const torch::Tensor & states() const { return _states; };

protected:
  TorchSize batch_size() const;

protected:
  ConstitutiveModel & _model;
  torch::Tensor _max_strain;
  torch::Tensor _strain_rate;

  torch::Tensor _forces;
  torch::Tensor _states;
};

/// Really simple torch csv writer
void write_csv(torch::Tensor tensor, std::string fname, std::string delimiter = ",");
