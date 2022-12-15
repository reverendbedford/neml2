#include "models/solid_mechanics/PlasticStrainRate.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
PlasticStrainRate::PlasticStrainRate(const std::string & name)
  : Model(name),
    hardening_rate(declareInputVariable<Scalar>({"state", "hardening_rate"})),
    plastic_flow_direction(declareInputVariable<SymR2>({"state", "plastic_flow_direction"})),
    plastic_strain_rate(declareOutputVariable<SymR2>({"state", "plastic_strain_rate"}))
{
  setup();
}

void
PlasticStrainRate::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  // Grab the consistency parameters
  auto gamma_dot = in.get<Scalar>(hardening_rate);
  auto Np = in.get<SymR2>(plastic_flow_direction);

  // Ep_dot = ep_dot * Np
  auto Ep_dot = gamma_dot * Np;

  // Set the output
  out.set(Ep_dot, plastic_strain_rate);

  if (dout_din)
  {
    TorchSize nbatch = in.batch_size();
    auto I = SymR2::identity_map().batch_expand(nbatch);

    dout_din->set(gamma_dot * I, plastic_strain_rate, plastic_flow_direction);
    dout_din->set(Np, plastic_strain_rate, hardening_rate);
  }
}
} // namespace neml2
