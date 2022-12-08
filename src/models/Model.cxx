#include "models/Model.h"

Model::Model(const std::string & name)
  : torch::nn::Module(name),
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
  auto [out, dout_din] = value_and_dvalue(in);
  return dout_din;
}

std::tuple<LabeledVector, LabeledMatrix>
Model::value_and_dvalue(LabeledVector in) const
{
  LabeledVector out(in.batch_size(), output());
  LabeledMatrix dout_din(out, in);
  set_value(in, out, &dout_din);
  return {out, dout_din};
}

void
Model::registerModel(const std::shared_ptr<Model> & model)
{
  input().merge(model->input());
  _registered_models.push_back(model);
  register_module(model->name(), model);
}
