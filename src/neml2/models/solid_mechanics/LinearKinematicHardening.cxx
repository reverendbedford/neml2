#include "neml2/models/solid_mechanics/LinearKinematicHardening.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
LinearKinematicHardening::LinearKinematicHardening(const std::string & name, Scalar H)
  : KinematicHardening(name),
    _H(register_parameter("kinematic_hardening_modulus", H))
{
}

void
LinearKinematicHardening::set_value(LabeledVector in,
                                    LabeledVector out,
                                    LabeledMatrix * dout_din) const
{
  // Map from equivalent plastic strain --> isotropic hardening
  auto g = _H * in.get<SymR2>(plastic_strain);

  // Set the output
  out.set(g, kinematic_hardening);

  if (dout_din)
  {
    // Derivative of the map equivalent plastic strain --> isotropic hardening
    auto dg_dep = _H * SymSymR4::init(SymSymR4::identity_sym).batch_expand(in.batch_size());

    // Set the output
    dout_din->set(dg_dep, kinematic_hardening, plastic_strain);
  }
}
} // namespace neml2
