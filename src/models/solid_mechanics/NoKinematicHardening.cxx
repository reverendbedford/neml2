#include "models/solid_mechanics/NoKinematicHardening.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
void
NoKinematicHardening::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  // Retrieve whatever we need from the input,
  // Here we need the cauchy stress
  auto cauchy = in.slice(0, "state").get<SymR2>("cauchy_stress");

  // Map from cauchy stress --> mandel stress
  // Well, without kinematic hardening mandel stress and cauchy stress coincide
  auto mandel = cauchy;

  // Set the output
  out.slice(0, "state").set(mandel, "mandel_stress");

  if (dout_din)
  {
    // Derivative of the map cauchy stress --> mandel stress
    auto dmandel_dcauchy = SymR2::identity_map().expand_batch(in.batch_size());

    // Set the output
    dout_din->block("state", "state").set(dmandel_dcauchy, "mandel_stress", "cauchy_stress");
  }
}
} // namespace neml2
