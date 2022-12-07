#include "models/solid_mechanics/YieldFunction.h"

YieldFunction::YieldFunction(const std::string & name)
  : Model(name)
{
  input().add<LabeledAxis>("state");
  input().subaxis("state").add<SymR2>("mandel_stress");

  output().add<LabeledAxis>("state");
  output().subaxis("state").add<Scalar>("yield_function");
  setup();
}

LabeledTensor<1, 3>
YieldFunction::d2value(LabeledVector in) const
{
  auto [dout_din, d2out_din2] = dvalue_and_d2value(in);
  return d2out_din2;
}

std::tuple<LabeledMatrix, LabeledTensor<1, 3>>
YieldFunction::dvalue_and_d2value(LabeledVector in) const
{
  LabeledMatrix dout_din(in.batch_size(), output(), in.axis(0));
  LabeledTensor<1, 3> d2out_din2(in.batch_size(), output(), in.axis(0), in.axis(0));
  set_dvalue(in, dout_din, &d2out_din2);
  return {dout_din, d2out_din2};
}
