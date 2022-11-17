#pragma once

#include "models/ConstitutiveModel.h"
#include "solvers/NonlinearSystem.h"

/// Base class for models defined with an implicit function
//  Arguments: {current_state, current_forces, past_state, past_forces}
class ImplicitFunctionModel : public NonlinearSystem, public ConstitutiveModel
{
public:
  // value defines the results
  // dvalue defines the gradient with respect to each argument

  /// Store the other state and do whatever other setup is required
  virtual void setup(StateInput input);

  /// Provide an initial guess at state_np1, default is state_n
  virtual State trial_state() const;

  /// Residual is just value
  virtual torch::Tensor residual(torch::Tensor x);

  /// Jacobian is the first part of dvalue
  virtual torch::Tensor jacobian(torch::Tensor x);

  /// Map a flat vector back to the state
  State map_state(torch::Tensor x) const;

  /// Gather the input needed to pass to value
  StateInput gather_input(torch::Tensor x) const;

protected:
  /// Info object needed to construct current state from a tensor
  StateInfo _state_info;
  /// Current driving forces and previous state
  //  TODO: consider using weak_ptrs or something
  std::shared_ptr<State> _forces_np1, _state_n, _forces_n;
};
