#pragma once

#include "state/State.h"
#include "state/StateInfo.h"

/// Decorator for an object that defines state
class StateProvider
{
public:
  /// Definition of the state variables
  virtual StateInfo state() const = 0;
  /// Setup the initial values of the state
  virtual void initial_state(State & state) const = 0;
};
