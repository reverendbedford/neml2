#include "neml2/models/solid_mechanics/VoceIsotropicHardening.h"

namespace neml2
{
VoceIsotropicHardening::VoceIsotropicHardening(const std::string & name, Scalar R, Scalar d)
  : IsotropicHardening(name),
    _R(register_parameter("saturated_hardening", R)),
    _d(register_parameter("saturation_rate", d))
{
}

void
VoceIsotropicHardening::set_value(LabeledVector in,
                                  LabeledVector out,
                                  LabeledMatrix * dout_din) const
{
  // Map from equivalent plastic strain --> isotropic hardening
  auto g = _R * (1.0 - exp(-_d * in.get<Scalar>(equivalent_plastic_strain)));

  // Set the output
  out.set(g, isotropic_hardening);

  if (dout_din)
  {
    // Derivative of the map equivalent plastic strain --> isotropic hardening
    auto dg_dep = _R * _d * exp(-_d * in.get<Scalar>(equivalent_plastic_strain));

    // Set the output
    dout_din->set(dg_dep, isotropic_hardening, equivalent_plastic_strain);
  }
}
} // namespace neml2
