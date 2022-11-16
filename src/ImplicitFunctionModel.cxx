#include "ImplicitFunctionModel.h"
#include "State.h"

void
ImplicitFunctionModel::setup(StateInput input)
{
  _forces_np1 = std::make_shared<State>(input[0]);
  _state_n = std::make_shared<State>(input[1]);
  _forces_n = std::make_shared<State>(input[2]);

  _state_info = _state_n->info();
}

State
ImplicitFunctionModel::trial_state() const
{
  // Default is just a copy of state_n
  return _state_n->clone();
}

torch::Tensor
ImplicitFunctionModel::residual(torch::Tensor x)
{
  return value(gather_input(x)).tensor();
}

torch::Tensor
ImplicitFunctionModel::jacobian(torch::Tensor x)
{
  return dvalue(gather_input(x))[0].tensor();
}

State
ImplicitFunctionModel::map_state(torch::Tensor x) const
{
  return State(_state_info, x);
}

StateInput
ImplicitFunctionModel::gather_input(torch::Tensor x) const
{
  return {map_state(x), *_forces_np1, *_state_n, *_forces_n};
}
