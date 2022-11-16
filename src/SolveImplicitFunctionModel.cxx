#include "SolveImplicitFunctionModel.h"
#include "State.h"
#include "StateInfo.h"

SolveImplicitFunctionModel::SolveImplicitFunctionModel(ImplicitFunctionModel & model,
                                                       const NonlinearSolver & solver)
  : _model(model),
    _solver(solver),
    _evaluated(false)
{
}

State
SolveImplicitFunctionModel::value(StateInput input)
{
  _model.setup(input);
  _last_x = _solver.solve(_model.trial_state().tensor(), _model);
  _evaluated = true;
  return _model.map_state(_last_x);
}

StateDerivativeOutput
SolveImplicitFunctionModel::dvalue(StateInput /*input*/)
{
  // You must first call value to solve the system or else this will
  // error

  // TODO: Check to see input is the same as that cached for last_x?
  if (!_evaluated)
    throw std::runtime_error("Implicit function Jacobian calculation requires "
                             "calling the model first to cache the converged "
                             "solution");

  StateDerivativeOutput partials = _model.dvalue(_model.gather_input(_last_x));

  // The first one is the Jacobian by our convention of putting the
  // implicit state first
  StateDerivative Ji = partials[0].inverse();

  // Loop over the remaining items and use the implicit function
  // theorem to calculate the appropriate total derivative
  StateDerivativeOutput res;
  for (size_t i = 1; i < partials.size(); i++)
    res.push_back(-Ji.chain(partials[i]));

  return res;
}

StateInfo
SolveImplicitFunctionModel::state() const
{
  // Take from the underlying model
  return _model.state();
}

StateInfo
SolveImplicitFunctionModel::forces() const
{
  // Take from the underlying model
  return _model.forces();
}

void
SolveImplicitFunctionModel::initial_state(State & state) const
{
  _model.initial_state(state);
}
