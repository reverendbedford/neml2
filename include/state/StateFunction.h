#pragma once

#include "state/State.h"
#include "state/StateDerivative.h"

// I'm not happy about pass by value here, even though
// the torch C++ API seems to encourage it.  I'm also
// not thrilled about returning a vector of results,
// some templating might be able to fix that.
typedef std::vector<State> StateInput;
typedef std::vector<StateDerivative> StateDerivativeOutput;

/// Class that maps some state input -> state output
class StateFunction
{
public:
  /// Which variables this object defines as output
  virtual StateInfo output() const = 0;

  // Really want these to be variadic in input
  /// The map between input -> output
  virtual State value(StateInput input) = 0;
  /// The derivative between input -> output
  virtual StateDerivativeOutput dvalue(StateInput input) = 0;
  // If you make a default version that uses AD...
};

/// Specialization of StateFunction with a single input
class SingleStateFunction : public StateFunction
{
public:
  /// Implementation of variable input map
  virtual State value(StateInput input);
  /// Implementation of variable input derivative
  virtual StateDerivativeOutput dvalue(StateInput input);

  /// Specialized single input interface
  virtual State value(State input) = 0;
  /// Specialized single input interface for derivative
  virtual StateDerivative dvalue(State input) = 0;
};
