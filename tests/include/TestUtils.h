#pragma once

#include "tensors/Scalar.h"
#include "state/State.h"
#include "state/StateDerivative.h"

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
