#pragma once

#include "ConstitutiveModel.h"
#include "ImplicitFunctionModel.h"
#include "NonlinearSolver.h"

/// Base class for materials defined with an implicit function
class SolveImplicitFunctionModel : public ConstitutiveModel
{
public:
  /// Construct with the solver to use to solve the system
  SolveImplicitFunctionModel(ImplicitFunctionModel & model, const NonlinearSolver & solver);

  /// Solve the system to advance the state
  virtual State value(StateInput input);

  /// Use the implicit function theorem to calculate the Jacobian
  virtual StateDerivativeOutput dvalue(StateInput input);

  /// Just get from the ImplicitFunctionModel
  virtual StateInfo state() const;

  /// Just get from the ImplicitFunctionModel
  virtual StateInfo forces() const;

  /// Just get from the ImplicitFunctionModel
  virtual void initial_state(State & state) const;

protected:
  /// The material model defined as an implicit function of state and forces
  ImplicitFunctionModel & _model;
  /// Nonlinear solver used to solve the system
  const NonlinearSolver & _solver;
  /// At least check to see if we've run value at least once...
  //  This is not a complete fix to the problem -- we'd have to
  //  hash the StateInput to figure that out
  bool _evaluated;
  /// The cached solution from the last time you called value
  torch::Tensor _last_x;
};
