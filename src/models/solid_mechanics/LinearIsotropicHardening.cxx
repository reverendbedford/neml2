#include "models/solid_mechanics/LinearIsotropicHardening.h"

namespace neml2
{
register_NEML2_object(LinearIsotropicHardening);

LinearIsotropicHardening::LinearIsotropicHardening(const std::string & name, Scalar s0, Scalar K)
  : IsotropicHardening(name),
    _s0(register_parameter("yield_stress", s0)),
    _K(register_parameter("hardening_modulus", K))
{
}

LinearIsotropicHardening::LinearIsotropicHardening(InputParameters & params)
  : IsotropicHardening(params),
    _s0(register_parameter("yield_stress", Scalar(params.param<Real>("s0")))),
    _K(register_parameter("hardening_modulus", Scalar(params.param<Real>("K"))))
{
}

void
LinearIsotropicHardening::set_value(LabeledVector in,
                                    LabeledVector out,
                                    LabeledMatrix * dout_din) const
{
  // Retrieve whatever we need from the input,
  // Here we need the equivalent plastic strain
  auto ep = in.slice(0, "state").get<Scalar>("equivalent_plastic_strain");

  // Map from equivalent plastic strain --> isotropic hardening
  auto h = _s0 + _K * ep;

  // Set the output
  out.slice(0, "state").set(h, "isotropic_hardening");

  if (dout_din)
  {
    // Derivative of the map equivalent plastic strain --> isotropic hardening
    auto dh_dep = _K.expand_batch(in.batch_size());

    // Set the output
    dout_din->block("state", "state")
        .set(dh_dep, "isotropic_hardening", "equivalent_plastic_strain");
  }
}
} // namespace neml2
