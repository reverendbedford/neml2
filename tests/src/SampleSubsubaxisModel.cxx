#include "SampleSubsubaxisModel.h"

using namespace neml2;

SampleSubsubaxisModel::SampleSubsubaxisModel(const std::string & name)
  : Model(name),
    foo(declareInputVariable<Scalar>("state", "foo")),
    bar(declareInputVariable<Scalar>("state", "substate", "bar")),
    baz(declareOutputVariable<Scalar>("state", "baz"))
{
  setup();
}

void
SampleSubsubaxisModel::set_value(LabeledVector in,
                                 LabeledVector out,
                                 LabeledMatrix * dout_din) const
{
  out.set(5 * Scalar(in(foo)) * Scalar(in(bar)), baz);

  if (dout_din)
  {
    dout_din->set(5 * Scalar(in(bar)), baz, foo);
    dout_din->set(5 * Scalar(in(foo)), baz, bar);
  }
}
