#include "models/solid_mechanics/AssociativeKinematicPlasticHardening.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
AssociativeKinematicPlasticHardening::AssociativeKinematicPlasticHardening(
    const std::string & name, const std::shared_ptr<YieldFunction> & f)
  : PlasticHardening(name),
    yield_function(*f),
    plastic_strain_rate(
        declareOutputVariable<SymR2>({"state", "internal_state", "plastic_strain_rate"}))
{
  register_model(f);
  setup();
}

void
AssociativeKinematicPlasticHardening::set_value(LabeledVector in,
                                       LabeledVector out,
                                       LabeledMatrix * dout_din) const
{
  // For associative flow,
  // ep_dot = - gamma_dot * df/dh
  TorchSize nbatch = in.batch_size();
  LabeledMatrix df_din(nbatch, yield_function.output(), yield_function.input());
  LabeledTensor<1, 3> d2f_din2(
      nbatch, yield_function.output(), yield_function.input(), yield_function.input());

  if (dout_din)
    std::tie(df_din, d2f_din2) = yield_function.dvalue_and_d2value(in);
  else
    df_din = yield_function.dvalue(in);

  auto gamma_dot = in.get<Scalar>(hardening_rate);
  auto df_dh =
      df_din.get<SymR2>(yield_function.yield_function, yield_function.kinematic_hardening);
  auto ep_dot = -gamma_dot * df_dh;

  out.set(ep_dot, plastic_strain_rate);

  if (dout_din)
  {
    // dep_dot/dh = -gamma_dot * d2f/dh2
    auto d2f_dh2 = d2f_din2.get<SymSymR4>(yield_function.yield_function,
                                        yield_function.kinematic_hardening,
                                        yield_function.kinematic_hardening);
    auto value = gamma_dot * d2f_dh2;

    dout_din->set(-df_dh, plastic_strain_rate, hardening_rate);
    dout_din->set(value, plastic_strain_rate, yield_function.mandel_stress);
    dout_din->set(
        -value, plastic_strain_rate, yield_function.kinematic_hardening);
  }
}
} // namespace neml2
