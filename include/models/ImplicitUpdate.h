#pragma once

#include "models/ImplicitModel.h"
#include "solvers/NonlinearSolver.h"

/// Update an implicit model by solving the underlying nonlinear system
class ImplicitUpdate : public Model
{
public:
  ImplicitUpdate(const std::string & name, ImplicitModel & model, NonlinearSolver & solver);

  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

protected:
  /// The implicit model to be updated
  ImplicitModel & _model;

  /// The nonlinear solver used to solve the nonlinear system
  NonlinearSolver & _solver;
};
