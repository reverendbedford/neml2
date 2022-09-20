#pragma once

#include "State.h"
#include "StateDerivative.h"
#include "StateInfo.h"

/// Abstract base class of all material models
class ConstitutiveModel {
 public:
  /// Update the state, return by value
  State update_state(const State & forces_np1, const State & state_n,
                     const State & forces_n);

  /// Update given a reference to a setup state object
  virtual void update(State & state_np1, const State & forces_np1,
                      const State & state_n, const State & forces_n) = 0;
  
  /// Define the state of this model
  virtual StateInfo state() const = 0;

  /// Define the driving forces for this model
  virtual StateInfo forces() const = 0;
};
