#include "models/solid_mechanics/PerzynaPlasticFlowRate.h"

PerzynaPlasticFlowRate::PerzynaPlasticFlowRate(const std::string & name, Scalar eta, Scalar n)
  : PlasticFlowRate(name),
    _eta(eta),
    _n(n)
{
}

void
PerzynaPlasticFlowRate::set_value(LabeledVector in,
                                  LabeledVector out,
                                  LabeledMatrix * dout_din) const
{
  // Grab the yield function
  auto f = in.slice(0, "state").get<Scalar>("yield_function");

  // Compute the Perzyna approximation of the yield surface
  Scalar Hf = torch::heaviside(f, torch::zeros_like(f));
  Scalar f_abs = torch::abs(f);
  Scalar ep_dot_m = torch::pow(f_abs / _eta, _n);
  Scalar ep_dot = ep_dot_m * Hf;

  // Set output
  out.slice(0, "state").set(ep_dot, "hardening_rate");

  if (dout_din)
  {
    // Compute the Perzyna approximation of the yield surface
    Scalar dep_dot_df = _n / f_abs * ep_dot * Hf;

    // Set output
    dout_din->block("state", "state").set(dep_dot_df, "hardening_rate", "yield_function");
  }
}
