#include "neml2/models/solid_mechanics/IsotropicMandelStress.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
void
IsotropicMandelStress::set_value(LabeledVector in,
                                 LabeledVector out,
                                 LabeledMatrix * dout_din) const
{
  // Isotropic mandel stress is just the Cauchy stress
  out.set(in.get<SymR2>(cauchy_stress), mandel_stress);

  if (dout_din)
  {
    // Derivative of the map cauchy stress --> mandel stress
    auto dmandel_dcauchy = SymR2::identity_map().batch_expand(in.batch_size());

    // Set the derivative
    dout_din->set(dmandel_dcauchy, mandel_stress, cauchy_stress);
  }
}
} // namespace neml2
