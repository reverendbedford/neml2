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
  params.set<bool>("use_AD_first_derivative") = false;
  params.set<bool>("use_AD_second_derivative") = false;
  return params;
}

Model::Model(const ParameterSet & params)
  : NEML2Object(params),
    _input(declareAxis()),
    _output(declareAxis()),
    _AD_1st_deriv(params.get<bool>("use_AD_first_derivative")),
    _AD_2nd_deriv(params.get<bool>("use_AD_second_derivative"))
{
  check_AD_limitation();
  for (const auto & var : params.get<std::vector<LabeledAxisAccessor>>("additional_outputs"))
    _additional_outputs.insert(var);
  setup();
}

void
Model::check_AD_limitation() const
{
  // AD_1st_deriv   AD_2nd_deriv   comment
  // true           true           okay, just slow
  // true           false          error, this is a weird case
  // false          true           okay
  // false          false          great, everything handcoded
  if (_AD_1st_deriv && !_AD_2nd_deriv)
    throw NEMLException("AD derivative is requested, but AD second derivative is not requested.");
}

void
Model::use_AD_derivatives(bool first, bool second)
{
  _AD_1st_deriv = first;
  _AD_2nd_deriv = second;
  check_AD_limitation();
}

void
Model::trace_parameters(const std::map<std::string, bool> & params)
{
  const auto & model_params = named_parameters(true);
  for (auto && [name, requires_grad] : params)
    model_params[name].requires_grad_(requires_grad);
}

void
Model::set_parameters(const std::map<std::string, torch::Tensor> & params)
{
  const auto & model_params = named_parameters(true);
  for (auto && [name, val] : params)
  {
    auto req_grad = model_params[name].requires_grad();
    model_params[name].requires_grad_(false);
    model_params[name].copy_(val);
    model_params[name].requires_grad_(req_grad);
  }
}

BatchTensor<1>
Model::dparam(const LabeledVector & out, const std::string & param) const
{
  const auto & outval = out.tensor();
  BatchTensor<1> pval = named_parameters(true)[param];

  BatchTensor<1> dout_dp(
      out.batch_size(), utils::add_shapes(outval.base_sizes(), pval.base_sizes()), out.options());

  if (pval.batch_sizes()[0] == 1)
  {
    for (TorchSize b = 0; b < outval.batch_sizes()[0]; b++)
      for (TorchSize i = 0; i < outval.base_sizes()[0]; i++)
      {
        BatchTensor<1> grad_outputs = torch::zeros_like(outval.index({b}));
        grad_outputs.index_put_({i}, 1.0);
        try
        {
          auto jac_row =
              torch::autograd::grad({outval.index({b})}, {pval}, {grad_outputs}, true)[0];
          dout_dp.index_put_({b, i, torch::indexing::Slice()}, jac_row);
        }
        catch (c10::Error &)
        {
          // This is aggravating: libTorch throws if the derivative is zero... but why?!
        }
      }
  }
  else
  {
    for (TorchSize i = 0; i < outval.base_sizes()[0]; i++)
    {
      BatchTensor<1> grad_outputs = torch::zeros_like(outval);
      grad_outputs.index_put_({torch::indexing::Ellipsis, i}, 1.0);
      try
      {
        auto jac_row = torch::autograd::grad({outval}, {pval}, {grad_outputs}, true)[0];
        dout_dp.base_index_put({i, torch::indexing::Slice()}, jac_row);
      }
      catch (c10::Error &)
      {
        // This is aggravating: libTorch throws if the derivative is zero... but why?!
      }
    }
  }

  return dout_dp;
}

LabeledVector
Model::value(const LabeledVector & in) const
{
  LabeledVector out(in.batch_size(), {&output()}, in.options());
  set_value(in, &out);
  return out;
}

LabeledMatrix
Model::dvalue(const LabeledVector & in) const
{
  if (_AD_1st_deriv)
    return std::get<1>(value_and_dvalue(in));

  LabeledMatrix dout_din(in.batch_size(), {&output(), &in.axis()}, in.options());
  set_value(in, nullptr, &dout_din);
  return dout_din;
}

LabeledTensor3D
Model::d2value(const LabeledVector & in) const
{
  if (_AD_2nd_deriv)
    return std::get<2>(value_and_dvalue_and_d2value(in));

  LabeledTensor3D d2out_din2(in.batch_size(), {&output(), &in.axis(), &in.axis()}, in.options());
  set_value(in, nullptr, nullptr, &d2out_din2);
  return d2out_din2;
}

std::tuple<LabeledVector, LabeledMatrix>
Model::value_and_dvalue(const LabeledVector & in) const
{
  LabeledVector out(in.batch_size(), {&output()}, in.options());
  LabeledMatrix dout_din(in.batch_size(), {&output(), &in.axis()}, in.options());

  if (_AD_1st_deriv)
  {
    // Set requires_grad to true if not already
    bool req_grad = in.tensor().requires_grad();
    in.tensor().requires_grad_();
    // Evaluate the model value
    set_value(in, &out);
    // Loop over rows to retrieve the derivatives
    for (TorchSize i = 0; i < out.tensor().base_sizes()[0]; i++)
    {
      BatchTensor<1> grad_outputs = torch::zeros_like(out.tensor());
      grad_outputs.index_put_({torch::indexing::Ellipsis, i}, 1.0);
      try
      {
        auto jac_row =
            torch::autograd::grad({out.tensor()}, {in.tensor()}, {grad_outputs}, true, true)[0];
        dout_din.tensor().base_index_put({i, torch::indexing::Slice()}, jac_row);
      }
      catch (c10::Error &)
      {
        // This is aggravating: libTorch throws if the derivative is zero... but why?!
      }
    }
    in.tensor().requires_grad_(req_grad);
  }
  else
    set_value(in, &out, &dout_din);

  return {out, dout_din};
}

std::tuple<LabeledMatrix, LabeledTensor3D>
Model::dvalue_and_d2value(const LabeledVector & in) const
{
  if (_AD_1st_deriv || _AD_2nd_deriv)
    return {std::get<1>(value_and_dvalue_and_d2value(in)),
            std::get<2>(value_and_dvalue_and_d2value(in))};

  LabeledMatrix dout_din(in.batch_size(), {&output(), &in.axis()}, in.options());
  LabeledTensor3D d2out_din2(in.batch_size(), {&output(), &in.axis(), &in.axis()}, in.options());
  set_value(in, nullptr, &dout_din, &d2out_din2);
  return {dout_din, d2out_din2};
}

std::tuple<LabeledVector, LabeledMatrix, LabeledTensor3D>
Model::value_and_dvalue_and_d2value(const LabeledVector & in) const
{
  LabeledVector out(in.batch_size(), {&output()}, in.options());
  LabeledMatrix dout_din(in.batch_size(), {&output(), &in.axis()}, in.options());
  LabeledTensor3D d2out_din2(in.batch_size(), {&output(), &in.axis(), &in.axis()}, in.options());

  if (_AD_2nd_deriv)
  {
    // Set requires_grad to true if not already
    bool req_grad = in.tensor().requires_grad();
    in.tensor().requires_grad_();
    if (_AD_1st_deriv)
    {
      set_value(in, &out);
      // Loop over rows to retrieve the derivatives
      for (TorchSize i = 0; i < out.tensor().base_sizes()[0]; i++)
      {
        BatchTensor<1> grad_outputs = torch::zeros_like(out.tensor());
        grad_outputs.index_put_({torch::indexing::Ellipsis, i}, 1.0);
        try
        {
          auto jac_row =
              torch::autograd::grad({out.tensor()}, {in.tensor()}, {grad_outputs}, true, true)[0];
          dout_din.tensor().base_index_put({i, torch::indexing::Slice()}, jac_row);
        }
        catch (c10::Error &)
        {
          // This is aggravating: libTorch throws if the derivative is zero... but why?!
        }
      }
    }
    else
      set_value(in, &out, &dout_din);
    // Loop over rows to retrieve the second derivatives
    for (TorchSize i = 0; i < dout_din.tensor().base_sizes()[0]; i++)
      for (TorchSize j = 0; j < dout_din.tensor().base_sizes()[1]; j++)
      {
        BatchTensor<1> grad_outputs = torch::zeros_like(dout_din.tensor());
        grad_outputs.index_put_({torch::indexing::Ellipsis, i, j}, 1.0);
        try
        {
          auto jac_row =
              torch::autograd::grad({dout_din.tensor()}, {in.tensor()}, {grad_outputs}, true)[0];
          d2out_din2.tensor().base_index_put({i, j, torch::indexing::Slice()}, jac_row);
        }
        catch (c10::Error &)
        {
          // This is aggravating: libTorch throws if the derivative is zero... but why?!
        }
      }
    in.tensor().requires_grad_(req_grad);
  }
  else
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
Model::cache_input(const LabeledVector & in)
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
