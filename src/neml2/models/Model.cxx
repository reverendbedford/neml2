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
  params.set<std::vector<LabeledAxisAccessor>>("additional_outputs");
  return params;
}

Model::Model(const ParameterSet & params)
  : NEML2Object(params),
    _input(declareAxis()),
    _output(declareAxis())
{
  for (const auto & var : params.get<std::vector<LabeledAxisAccessor>>("additional_outputs"))
    _additional_outputs.insert(var);
  setup();
}

LabeledVector
Model::value(LabeledVector in) const
{
  LabeledVector out(in.batch_size(), {&output()}, in.options());
  set_value(in, &out);
  return out;
}

LabeledMatrix
Model::dvalue(LabeledVector in) const
{
  LabeledMatrix dout_din(in.batch_size(), {&output(), &input()}, in.options());
  set_value(in, nullptr, &dout_din);
  return dout_din;
}

LabeledTensor3D
Model::d2value(LabeledVector in) const
{
  LabeledTensor3D d2out_din2(in.batch_size(), {&output(), &input(), &input()}, in.options());
  set_value(in, nullptr, nullptr, &d2out_din2);
  return d2out_din2;
}

std::tuple<LabeledVector, LabeledMatrix>
Model::value_and_dvalue(LabeledVector in) const
{
  LabeledVector out(in.batch_size(), {&output()}, in.options());
  LabeledMatrix dout_din(in.batch_size(), {&output(), &input()}, in.options());
  set_value(in, &out, &dout_din);
  return {out, dout_din};
}

std::tuple<LabeledMatrix, LabeledTensor3D>
Model::dvalue_and_d2value(LabeledVector in) const
{
  LabeledMatrix dout_din(in.batch_size(), {&output(), &input()}, in.options());
  LabeledTensor3D d2out_din2(in.batch_size(), {&output(), &input(), &input()}, in.options());
  set_value(in, nullptr, &dout_din, &d2out_din2);
  return {dout_din, d2out_din2};
}

std::tuple<LabeledVector, LabeledMatrix, LabeledTensor3D>
Model::value_and_dvalue_and_d2value(LabeledVector in) const
{
  LabeledVector out(in.batch_size(), {&output()}, in.options());
  LabeledMatrix dout_din(in.batch_size(), {&output(), &input()}, in.options());
  LabeledTensor3D d2out_din2(in.batch_size(), {&output(), &input(), &input()}, in.options());
  set_value(in, &out, &dout_din, &d2out_din2);
  return {out, dout_din, d2out_din2};
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

  _registered_models.push_back(model.get());

  // torch bookkeeping
  register_module(model->name(), model);
}

void
Model::cache_input(LabeledVector in)
{
  _cached_in = in.clone();
}

void
Model::advance_step()
{
  for (auto model : registered_models())
    model->advance_step();
}

void
Model::set_residual(BatchTensor<1> x, BatchTensor<1> * r, BatchTensor<1> * J) const
{
  const auto options = x.options();
  const auto nbatch = x.batch_sizes()[0];

  LabeledVector in(nbatch, {&input()}, options);

  // Fill in the current trial state and cached (fixed) forces, old forces, old state
  in.fill(_cached_in);
  in.set(x, "state");

  // Let's try to be as efficient as possible by considering all the cases!
  if (r && !J)
  {
    auto out = value(in);
    r->copy_(out("residual"));
  }
  else if (!r && J)
  {
    auto dout_din = dvalue(in);
    J->copy_(dout_din("residual", "state"));
  }
  else if (r && J)
  {
    auto [out, dout_din] = value_and_dvalue(in);
    r->copy_(out("residual"));
    J->copy_(dout_din("residual", "state"));
  }
}
} // namespace neml2
