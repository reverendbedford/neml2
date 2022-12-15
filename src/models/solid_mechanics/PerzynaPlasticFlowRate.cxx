#include "models/solid_mechanics/PerzynaPlasticFlowRate.h"

namespace neml2
{
PerzynaPlasticFlowRate::PerzynaPlasticFlowRate(const std::string & name, Scalar eta, Scalar n)
  : PlasticFlowRate(name),
    _eta(register_parameter("reference_flow_stress", eta)),
    _n(register_parameter("flow_rate_exponent", n))
{
}

void
PerzynaPlasticFlowRate::set_value(LabeledVector in,
                                  LabeledVector out,
                                  LabeledMatrix * dout_din) const
{
  // Grab the yield function
  auto f = in.slice("state").get<Scalar>("yield_function");

  // Compute the Perzyna approximation of the yield surface
  Scalar Hf = torch::heaviside(f, torch::zeros_like(f));
  Scalar f_abs = torch::abs(f);
  Scalar gamma_dot_m = torch::pow(f_abs / _eta, _n);
  Scalar gamma_dot = gamma_dot_m * Hf;

  // Set output
  out.slice("state").set(gamma_dot, "hardening_rate");

  if (dout_din)
  {
    // Compute the Perzyna approximation of the yield surface
    Scalar dgamma_dot_df = _n / f_abs * gamma_dot * Hf;

    // Set output
    dout_din->block("state", "state").set(dgamma_dot_df, "hardening_rate", "yield_function");
  }
}
} // namespace neml2
