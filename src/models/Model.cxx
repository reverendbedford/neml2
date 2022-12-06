#include "models/Model.h"

Model::Model(const std::string & name)
  : _name(name),
    _input(declareAxis()),
    _output(declareAxis())
{
  setup();
}

LabeledVector
Model::value(LabeledVector in) const
{
  LabeledVector out(in.batch_size(), output());
  set_value(in, out);
  return out;
}

LabeledMatrix
Model::dvalue(LabeledVector in) const
{
  LabeledVector out(in.batch_size(), output());
  LabeledMatrix dout_din(in.batch_size(), output(), input());
  set_value(in, out, &dout_din);
  return dout_din;
}
