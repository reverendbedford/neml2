#include "models/solid_mechanics/PerzynaInelasticFlowRate.h"

PerzynaInelasticFlowRate::PerzynaInelasticFlowRate(Scalar eta,
                                                   Scalar n,
                                                   const YieldSurface & surface,
                                                   HardeningMap & map)
  : _eta(eta),
    _n(n),
    _surface(surface),
    _map(map)
{
}

State
PerzynaInelasticFlowRate::value(State input)
{
  auto res = State::same_batch(output(), input);
  auto f = _surface.f(_map.value(input));

  res.set<Scalar>("flow_rate",
                  torch::pow(torch::abs(f) / _eta, _n) * torch::heaviside(f, torch::zeros_like(f)));
  return res;
}

StateDerivative
PerzynaInelasticFlowRate::dvalue(State input)
{
  auto f = _surface.f(_map.value(input));
  auto df = _surface.df_ds(_map.value(input)).promote_left("flow_rate").chain(_map.dvalue(input));
  Scalar prefactor = _n / _eta * torch::pow(torch::abs(f) / _eta, _n - 1.0) *
                     torch::heaviside(f, torch::zeros_like(f));

  return df.scalar_product(prefactor);
}
