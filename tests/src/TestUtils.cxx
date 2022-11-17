#include "TestUtils.h"

using namespace torch::indexing;

State
scalar_derivative(std::function<Scalar(State)> func, const State & x, Real eps, Real aeps)
{
  Scalar y0 = func(x);
  State result = x.clone();

  for (TorchSize i = 0; i < x.tensor().sizes()[1]; i++)
  {
    auto dx = eps * torch::abs(x.tensor().index({Slice(), i}));
    dx.index_put_({dx < aeps}, aeps);

    State x1 = State(x.info(), x.tensor().clone());
    x1.tensor().index_put_({Slice(), i}, x1.tensor().index({Slice(), i}) + dx);

    auto y1 = func(x1);
    result.tensor().index_put_({Slice(), i}, (y1 - y0).index({Slice(), 0}) / dx);
  }

  return result;
}

StateDerivative
state_derivative(std::function<State(State)> func, const State & x, Real eps, Real aeps)
{
  State y0 = func(x);
  StateDerivative result(y0, x);

  for (TorchSize i = 0; i < x.tensor().sizes()[1]; i++)
  {
    auto dx = eps * torch::abs(x.tensor().index({Slice(), i}));
    dx.index_put_({dx < aeps}, aeps);

    State x1 = State(x.info(), x.tensor().clone());
    x1.tensor().index_put_({Slice(), i}, x1.tensor().index({Slice(), i}) + dx);

    auto y1 = func(x1);
    result.tensor().index_put_({"...", i}, (y1.tensor() - y0.tensor()) / dx.unsqueeze(-1));
  }

  return result;
}

std::vector<StateDerivative>
state_derivatives(std::function<State(std::vector<State>)> func,
                  std::vector<State> x,
                  Real eps,
                  Real aeps)
{
  std::vector<StateDerivative> res;
  for (size_t i = 0; i < x.size(); i++)
  {
    res.push_back(state_derivative(
        [i, func, x](State y) -> State
        {
          std::vector<State> inp;
          for (size_t j = 0; j < x.size(); j++)
          {
            if (i == j)
              inp.push_back(y);
            else
              inp.push_back(x[j]);
          }
          return func(inp);
        },
        x[i],
        eps,
        aeps));
  }
  return res;
}
