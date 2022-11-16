#pragma once

#include "types.h"

class Scalar;
class State;
class StateDerivative;

namespace utils
{
constexpr double sqrt2 = 1.4142135623730951;

inline constexpr double
mandelFactor(TorchSize i)
{
  return i < 3 ? 1.0 : sqrt2;
}

/// Derivative of a BatchedScalar(State) function
State scalar_derivative(std::function<Scalar(State)> func,
                        const State & x,
                        Real eps = 1e-6,
                        Real aeps = 1e-6);

/// Derivative of a State(State) function
StateDerivative state_derivative(std::function<State(State)> func,
                                 const State & x,
                                 Real eps = 1e-6,
                                 Real aeps = 1e-6);

/// List of derivatives of a State(StateInput) function
std::vector<StateDerivative> state_derivatives(std::function<State(std::vector<State>)> func,
                                               std::vector<State> x,
                                               Real eps = 1e-6,
                                               Real aeps = 1e-6);
}
