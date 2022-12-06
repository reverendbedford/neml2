#pragma once

#include "models/Model.h"
#include "solvers/NonlinearSystem.h"

/// Base class for all implicit models defined in terms of residual
class ImplicitModel : public Model, public NonlinearSystem
{
public:
  using Model::Model;

  enum Stage
  {
    SOLVING,
    UPDATING
  };
  static ImplicitModel::Stage stage;

  virtual BatchTensor<1> initial_guess(LabeledVector in) const;

  void cache_input(LabeledVector in);

protected:
  /// Cached input while solving this implicit model
  LabeledVector _cached_in;
};
