// Copyright 2023, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "neml2/models/Model.h"

namespace neml2
{
Model::Stage Model::stage = UPDATING;

ParameterSet
Model::expected_params()
{
  ParameterSet params = NEML2Object::expected_params();
  params.set<std::vector<std::vector<std::string>>>("additional_outputs") =
      std::vector<std::vector<std::string>>();
  return params;
}

Model::Model(const ParameterSet & params)
  : NEML2Object(params),
    _input(declareAxis()),
    _output(declareAxis())
{
  for (const auto & var : params.get<std::vector<std::vector<std::string>>>("additional_outputs"))
    _additional_outputs.insert(LabeledAxisAccessor{var});
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
    _consumed_vars.insert(merged_vars.begin(), merged_vars.end());
  }

  _registered_models.push_back(model);

  // torch bookkeeping
  register_module(model->name(), model);
}

BatchTensor<1>
Model::initial_guess(LabeledVector in, LabeledVector guess) const
{
  guess.fill(in.slice("old_state"));
  return guess.tensor();
}

void
Model::cache_input(LabeledVector in)
{
  _cached_in = in.clone();
}

void
Model::set_residual(BatchTensor<1> x, BatchTensor<1> r, BatchTensor<1> * J) const
{
  TorchSize nbatch = x.batch_sizes()[0];
  LabeledVector in(nbatch, input());
  LabeledVector out(nbatch, output());

  // Fill in the current trial state and cached (fixed) forces, old forces, old state
  in.fill(_cached_in);
  in.set(x, "state");

  if (J)
  {
    LabeledMatrix dout_din(out, in);
    set_value(in, out, &dout_din);
    J->copy_(dout_din("residual", "state"));
  }
  else
    set_value(in, out);

  r.copy_(out("residual"));
}
} // namespace neml2
