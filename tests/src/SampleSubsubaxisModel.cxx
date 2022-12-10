#include "SampleSubsubaxisModel.h"

using namespace neml2;

SampleSubsubaxisModel::SampleSubsubaxisModel(const std::string & name)
  : Model(name),
    _foo(declareVariable<Scalar>(input(), "state", "foo")),
    _bar(declareVariable<Scalar>(input(), "state", "substate", "bar")),
    _baz(declareVariable<Scalar>(output(), "state", "baz"))
{
  setup();
}

void
SampleSubsubaxisModel::set_value(LabeledVector in,
                                 LabeledVector out,
                                 LabeledMatrix * dout_din) const
{
  out.set(5 * Scalar(in(_foo)) * Scalar(in(_bar)), _baz);

  if (dout_din)
  {
    dout_din->set(5 * Scalar(in(_bar)), _baz, _foo);
    dout_din->set(5 * Scalar(in(_foo)), _baz, _bar);
  }
}
