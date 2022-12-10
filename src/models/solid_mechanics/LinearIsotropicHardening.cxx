#include "models/solid_mechanics/LinearIsotropicHardening.h"

namespace neml2
{
LinearIsotropicHardening::LinearIsotropicHardening(const std::string & name, Scalar s0, Scalar K)
  : IsotropicHardening(name),
    _s0(register_parameter("yield_stress", s0)),
    _K(register_parameter("hardening_modulus", K))
{
}

void
LinearIsotropicHardening::set_value(LabeledVector in,
                                    LabeledVector out,
                                    LabeledMatrix * dout_din) const
{
  // Map from equivalent plastic strain --> isotropic hardening
  auto g = _s0 + _K * in.get<Scalar>(_ep_idx);

  // Set the output
  out.set(g, _g_idx);

  if (dout_din)
  {
    // Derivative of the map equivalent plastic strain --> isotropic hardening
    auto dg_dep = _K.batch_expand(in.batch_size());

    // Set the output
    dout_din->set(dg_dep, _g_idx, _ep_idx);
  }
}
} // namespace neml2
