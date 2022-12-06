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
  LabeledMatrix dout_din(in.batch_size(), output(), input());
  LabeledTensor<1, 3> d2out_din2(in.batch_size(), output(), input(), input());
  set_dvalue(in, dout_din, &d2out_din2);
  return d2out_din2;
}
