#include "models/solid_mechanics/LinearIsotropicHardening.h"

namespace neml2
{
LinearIsotropicHardening::LinearIsotropicHardening(const std::string & name, Scalar K)
  : IsotropicHardening(name),
    _K(register_parameter("hardening_modulus", K))
{
}

void
LinearIsotropicHardening::set_value(LabeledVector in,
                                    LabeledVector out,
                                    LabeledMatrix * dout_din) const
{
  // Map from equivalent plastic strain --> isotropic hardening
  auto g = _K * in.get<Scalar>(equivalent_plastic_strain);

  // Set the output
  out.set(g, isotropic_hardening);

  if (dout_din)
  {
    // Derivative of the map equivalent plastic strain --> isotropic hardening
    auto dg_dep = _K.batch_expand(in.batch_size());

    // Set the output
    dout_din->set(dg_dep, isotropic_hardening, equivalent_plastic_strain);
  }
}
} // namespace neml2
