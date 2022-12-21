#include "models/solid_mechanics/ChabochePlasticHardening.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
ChabochePlasticHardening::ChabochePlasticHardening(const std::string & name,
                                                   Scalar C,
                                                   Scalar g,
                                                   Scalar A,
                                                   Scalar a,
                                                   const std::string backstress_suffix)
  : PlasticHardening(name),
    backstress(
        declareInputVariable<SymR2>({"state", "internal_state", "backstress" + backstress_suffix})),
    flow_direction(declareInputVariable<SymR2>({"state", "plastic_flow_direction"})),
    backstress_rate(declareOutputVariable<SymR2>(
        {"state", "internal_state", "backstress" + backstress_suffix + "_rate"})),
    _C(register_parameter("chaboche_C" + backstress_suffix, C)),
    _g(register_parameter("chaboche_gamma" + backstress_suffix, g)),
    _A(register_parameter("chaboche_recovery_prefactor" + backstress_suffix, A)),
    _a(register_parameter("chaboche_recovery_exponent" + backstress_suffix, a))
{
  setup();
}

void
ChabochePlasticHardening::set_value(LabeledVector in,
                                    LabeledVector out,
                                    LabeledMatrix * dout_din) const
{
  // Our backstress
  SymR2 X = in.get<SymR2>(backstress);

  // gamma_dot
  Scalar g = in.get<Scalar>(hardening_rate);

  // Current flow direction
  SymR2 n = in.get<SymR2>(flow_direction);

  // Value of the effective stress for recovery
  auto eff = X.norm(); // Should already be deviatoric

  // Finally we can start assembling the model
  // Proportional to plastic strain rate
  auto g_term = 2.0 / 3.0 * _C * n - _g * X;
  // Static recovery
  auto s_term = -_A * eff.pow(_a - 1.0) * X;
  // Sum and set total
  auto total = g_term * g + s_term;
  out.set(total, backstress_rate);

  if (dout_din)
  {
    auto Y = X / (eff + EPS);

    // Plastic strain rate derivative
    dout_din->set(g_term, backstress_rate, hardening_rate);

    // Useful identity...
    auto I = SymSymR4::init(SymSymR4::identity_sym).batch_expand(in.batch_size());

    // Flow direction derivative
    dout_din->set(2.0 / 3.0 * _C * I * g, backstress_rate, flow_direction);

    // Backstress derivative
    dout_din->set(-torch::Tensor(_g * I * g) -
                      torch::Tensor(_A * (_a - 1.0) * (eff + EPS).pow(_a - 2.0) * X.outer(Y)) -
                      torch::Tensor(_A * eff.pow(_a - 1.0) * I),
                  backstress_rate,
                  backstress);
  }
}

} // namespace neml2
