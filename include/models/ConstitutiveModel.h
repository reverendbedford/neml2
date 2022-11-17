#pragma once

#include <torch/torch.h>

#include "state/StateProvider.h"
#include "state/StateFunction.h"

/// Abstract base class of all material models
class ConstitutiveModel : public StateFunction, public StateProvider
{
public:
  /// Alias for value
  State state_update(const State & forces_np1, const State & state_n, const State & forces_n);

  /// Alias for dvalue
  StateDerivativeOutput
  linearized_state_update(const State & forces_np1, const State & state_n, const State & forces_n);

  /// Alias for state, as we are mapping state -> state
  virtual StateInfo output() const;

  /// Define the driving forces for this model
  virtual StateInfo forces() const = 0;
};
