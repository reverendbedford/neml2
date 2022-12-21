#include "neml2/models/Model.h"

namespace neml2
{
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
Model::register_model(std::shared_ptr<Model> model, bool merge_input)
{
  if (merge_input)
  {
    // Additional inputs from the the registered model
    auto merged_vars = input().merge(model->input());
    _consumed_vars.insert(_consumed_vars.end(), merged_vars.begin(), merged_vars.end());
  }

  _registered_models.push_back(model);

  // torch bookkeeping
  register_module(model->name(), model);
}
} // namespace neml2
