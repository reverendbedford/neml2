#include "models/solid_mechanics/PlasticStrainRate.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
PlasticStrainRate::PlasticStrainRate(const std::string & name)
  : Model(name)
{
  input().add<LabeledAxis>("state");
  input().subaxis("state").add<Scalar>("hardening_rate");
  input().subaxis("state").add<SymR2>("plastic_flow_direction");

  output().add<LabeledAxis>("state");
  output().subaxis("state").add<SymR2>("plastic_strain_rate");

  setup();
}

void
PlasticStrainRate::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  // Grab the consistency parameters
  auto gamma_dot = in.slice(0, "state").get<Scalar>("hardening_rate");
  auto Np = in.slice(0, "state").get<SymR2>("plastic_flow_direction");

  // Ep_dot = ep_dot * Np
  auto Ep_dot = gamma_dot * Np;

  // Set the output
  out.slice(0, "state").set(Ep_dot, "plastic_strain_rate");

  if (dout_din)
  {
    TorchSize nbatch = in.batch_size();
    auto I = SymR2::identity_map().batch_expand(nbatch);

    dout_din->block("state", "state")
        .set(gamma_dot * I, "plastic_strain_rate", "plastic_flow_direction");
    dout_din->block("state", "state").set(Np, "plastic_strain_rate", "hardening_rate");
  }
}
} // namespace neml2
