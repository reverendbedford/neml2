#include "models/solid_mechanics/ChabochePlasticHardening.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
ChabochePlasticHardening::ChabochePlasticHardening(
    const std::string & name, Scalar C, Scalar g,
    Scalar A, Scalar a,
    const std::shared_ptr<YieldFunction> & f,
    const std::string backstress_suffix) :
    PlasticHardening(name),
    yield_function(*f),
    backstress(declareInputVariable<SymR2>({"state", "internal_state", "backstress" + backstress_suffix})),
    backstress_rate(declareOutputVariable<SymR2>({"state", "internal_state", "backstress" + backstress_suffix + "_rate"})),
    _C(register_parameter("chaboche_C" + backstress_suffix, C)),
    _g(register_parameter("chaboche_gamma" + backstress_suffix, g)),
    _A(register_parameter("chaboche_recovery_prefactor" + backstress_suffix, A)),
    _a(register_parameter("chaboche_recovery_exponent" + backstress_suffix, a))
{
  register_model(f);
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

  // Going to need the yield function derivative and 
  // second derivative
  TorchSize nbatch = in.batch_size();
  LabeledMatrix df_din(nbatch, yield_function.output(), yield_function.input());
  LabeledTensor<1, 3> d2f_din2(
      nbatch, yield_function.output(), yield_function.input(), yield_function.input());

  if (dout_din)
    std::tie(df_din, d2f_din2) = yield_function.dvalue_and_d2value(in);
  else
    df_din = yield_function.dvalue(in);

  // Also going to need the value of the stress measure, *given the
  // backstress as input*
  // First retrieve the hardening variables
  LabeledVector sm_input(nbatch, yield_function.stress_measure.input());
  sm_input.slice("state").set(X, "overstress");
  LabeledVector sm_value = yield_function.stress_measure.value(sm_input);
  LabeledMatrix sm_derivative;

  if (dout_din)
    sm_derivative = yield_function.stress_measure.dvalue(sm_input);
  
  // Finally we can start assembling the model
  auto n = df_din.get<SymR2>(yield_function.yield_function,
                             yield_function.mandel_stress);
  auto eff = sm_value.slice("state").get<Scalar>("stress_measure");
  auto g_term = 2.0 / 3.0 * _C * n - sqrt(2.0/3.0) * _g * X; 
  auto s_term = -sqrt(3.0/2.0) * _A * eff.pow(_a - 1.0) * X;
  auto v = g_term * g + s_term;
  out.set(v, backstress_rate);

  if (dout_din)
  {
    auto d2f_ds2 = d2f_din2.get<SymSymR4>(yield_function.yield_function,
                                          yield_function.mandel_stress,
                                          yield_function.mandel_stress);
    auto d2f_dk2 = d2f_din2.get<SymSymR4>(yield_function.yield_function,
                                          yield_function.mandel_stress,
                                          yield_function.kinematic_hardening);
    auto Y = sm_derivative.slice(0,"state").slice(1,"state").get<SymR2>("stress_measure", "overstress");

    // Plastic strain rate derivative
    dout_din->set(g_term, backstress_rate, hardening_rate);
    
    // Mandel stress derivative
    dout_din->set(2.0 / 3.0 * _C * d2f_ds2 * g, backstress_rate, 
                  yield_function.mandel_stress);

    // Kinematic hardening derivative
    dout_din->set(2.0/3.0 * _C * d2f_dk2 * g, backstress_rate,
                  yield_function.kinematic_hardening);

    // Backstress derivative
    dout_din->set(
        -torch::Tensor(sqrt(2.0/3.0) * _g * SymSymR4::init(SymSymR4::identity_sym).batch_expand(nbatch) * g)
        -torch::Tensor(sqrt(3.0/2.0) * _A * (_a - 1.0) * eff.pow(_a - 2.0) * X.outer(Y))
        -torch::Tensor(sqrt(3.0/2.0) * _A * eff.pow(_a - 1.0) * SymSymR4::init(SymSymR4::identity_sym).batch_expand(nbatch)),
                  backstress_rate, backstress);
  }

}



} // namespace neml2
