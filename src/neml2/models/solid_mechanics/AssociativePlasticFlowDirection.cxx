#include "neml2/models/solid_mechanics/AssociativePlasticFlowDirection.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
AssociativePlasticFlowDirection::AssociativePlasticFlowDirection(
    const std::string & name, const std::shared_ptr<YieldFunction> & f)
  : PlasticFlowDirection(name),
    yield_function(*f)
{
  register_model(f);
  setup();
}

void
AssociativePlasticFlowDirection::set_value(LabeledVector in,
                                           LabeledVector out,
                                           LabeledMatrix * dout_din) const
{
  // For associative flow, the flow direction is Np = df/dM
  TorchSize nbatch = in.batch_size();
  LabeledMatrix df_din(nbatch, yield_function.output(), yield_function.input());
  LabeledTensor<1, 3> d2f_din2(
      nbatch, yield_function.output(), yield_function.input(), yield_function.input());

  if (dout_din)
    std::tie(df_din, d2f_din2) = yield_function.dvalue_and_d2value(in);
  else
    df_din = yield_function.dvalue(in);

  auto df_dmandel = df_din.get<SymR2>(yield_function.yield_function, yield_function.mandel_stress);

  out.set(df_dmandel, plastic_flow_direction);

  if (dout_din)
  {
    // dNp/dM = d2f/dM2
    auto d2f_dmandel2 = d2f_din2.get<SymSymR4>(
        yield_function.yield_function, yield_function.mandel_stress, yield_function.mandel_stress);

    dout_din->set(d2f_dmandel2, plastic_flow_direction, yield_function.mandel_stress);
  }
}
} // namespace neml2
