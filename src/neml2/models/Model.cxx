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

OptionSet
Model::expected_options()
{
  OptionSet options = NEML2Object::expected_options();
  options.set<std::vector<LabeledAxisAccessor>>("additional_outputs");
  options.set<bool>("use_AD_first_derivative") = false;
  options.set<bool>("use_AD_second_derivative") = false;
  return options;
}

Model::Model(const OptionSet & options)
  : NEML2Object(options),
    _input(declareAxis()),
    _output(declareAxis()),
    _AD_1st_deriv(options.get<bool>("use_AD_first_derivative")),
    _AD_2nd_deriv(options.get<bool>("use_AD_second_derivative"))
{
  check_AD_limitation();
  for (const auto & var : options.get<std::vector<LabeledAxisAccessor>>("additional_outputs"))
    _additional_outputs.insert(var);
  setup();
}

void
Model::check_AD_limitation() const
{
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

BatchTensor
Model::dparam(const BatchTensor & out, const BatchTensor & p) const
{
  neml_assert(p.batch_dim() == 0 || out.batch_sizes() == p.batch_sizes(),
              "If the parameter is batched, its batch shape must be the same as the batch shape "
              "of the output. However, the batch shape of the parameter is ",
              p.batch_sizes(),
              ", and the batch shape of the output is ",
              out.batch_sizes());

  // flatten out to handle arbitrarily shaped output
  auto outf = BatchTensor(
      out.reshape(utils::add_shapes(out.batch_sizes(), utils::storage_size(out.base_sizes()))),
      out.batch_dim());

  neml_assert_dbg(outf.base_dim() == 1, "Flattened output must be flat.");

  auto doutf_dp = BatchTensor::empty(
      outf.batch_sizes(), utils::add_shapes(outf.base_sizes(), p.base_sizes()), outf.options());

  for (TorchSize i = 0; i < outf.base_sizes()[0]; i++)
  {
    auto G = BatchTensor::zeros_like(outf);
    G.index_put_({torch::indexing::Ellipsis, i}, 1.0);
    auto doutfi_dp = torch::autograd::grad({outf},
                                           {p},
                                           {G},
                                           /*retain_graph=*/true,
                                           /*create_graph=*/false,
                                           /*allow_unused=*/false)[0];
    if (doutfi_dp.defined())
      doutf_dp.base_index_put({i, torch::indexing::Ellipsis}, doutfi_dp);
  }

  // reshape the derivative back to the correct shape
  auto dout_dp = BatchTensor(
      doutf_dp.reshape(utils::add_shapes(out.batch_sizes(), out.base_sizes(), p.base_sizes())),
      out.batch_dim());

  // factor to account for broadcasting
  Real factor = p.batch_dim() == 0 ? utils::storage_size(out.batch_sizes()) : 1;

  return dout_dp / factor;
}

LabeledVector
Model::value(const LabeledVector & in) const
{
  auto out = LabeledVector::zeros(in.batch_sizes(), {&output()}, in.options());
  set_value(in, &out);
  return out;
}

LabeledMatrix
Model::dvalue(const LabeledVector & in) const
{
  if (_AD_1st_deriv)
    return std::get<1>(value_and_dvalue(in));

  auto dout_din = LabeledMatrix::zeros(in.batch_sizes(), {&output(), &in.axis()}, in.options());
  if (implicit())
  {
    auto out = LabeledVector::zeros(in.batch_sizes(), {&output()}, in.options());
    set_value(in, &out, &dout_din);
  }
  else
    set_value(in, nullptr, &dout_din);
  return dout_din;
}

LabeledTensor3D
Model::d2value(const LabeledVector & in) const
{
  neml_assert_dbg(
      !implicit(), name(), " is an implicit model and does not provide second derivatives.");

  if (_AD_2nd_deriv)
    return std::get<2>(value_and_dvalue_and_d2value(in));

  auto d2out_din2 =
      LabeledTensor3D::zeros(in.batch_sizes(), {&output(), &in.axis(), &in.axis()}, in.options());
  set_value(in, nullptr, nullptr, &d2out_din2);
  return d2out_din2;
}

std::tuple<LabeledVector, LabeledMatrix>
Model::value_and_dvalue(const LabeledVector & in) const
{
  auto out = LabeledVector::zeros(in.batch_sizes(), {&output()}, in.options());
  auto dout_din = LabeledMatrix::zeros(in.batch_sizes(), {&output(), &in.axis()}, in.options());

  if (_AD_1st_deriv)
  {
    // Set requires_grad to true if not already
    bool req_grad = in.tensor().requires_grad();
    in.tensor().requires_grad_();
    // Evaluate the model value
    set_value(in, &out);
    // Loop over rows to retrieve the derivatives
    for (TorchSize i = 0; i < out.base_sizes()[0]; i++)
    {
      auto grad_outputs = BatchTensor::zeros_like(out);
      grad_outputs.index_put_({torch::indexing::Ellipsis, i}, 1.0);
      out.tensor().requires_grad_();
      auto jac_row = torch::autograd::grad({out},
                                           {in},
                                           {grad_outputs},
                                           /*retain_graph=*/true,
                                           /*create_graph=*/false,
                                           /*allow_unused=*/true)[0];
      if (jac_row.defined())
        dout_din.base_index_put({i, torch::indexing::Slice()}, jac_row);
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
  neml_assert_dbg(
      !implicit(), name(), " is an implicit model and does not provide second derivatives.");

  if (_AD_1st_deriv || _AD_2nd_deriv)
  {
    auto [val, dval, d2val] = value_and_dvalue_and_d2value(in);
    return {dval, d2val};
  }

  auto dout_din = LabeledMatrix::zeros(in.batch_sizes(), {&output(), &in.axis()}, in.options());
  auto d2out_din2 =
      LabeledTensor3D::zeros(in.batch_sizes(), {&output(), &in.axis(), &in.axis()}, in.options());
  set_value(in, nullptr, &dout_din, &d2out_din2);
  return {dout_din, d2out_din2};
}

std::tuple<LabeledVector, LabeledMatrix, LabeledTensor3D>
Model::value_and_dvalue_and_d2value(const LabeledVector & in) const
{
  neml_assert_dbg(
      !implicit(), name(), " is an implicit model and does not provide second derivatives.");

  auto out = LabeledVector::zeros(in.batch_sizes(), {&output()}, in.options());
  auto dout_din = LabeledMatrix::zeros(in.batch_sizes(), {&output(), &in.axis()}, in.options());
  auto d2out_din2 =
      LabeledTensor3D::zeros(in.batch_sizes(), {&output(), &in.axis(), &in.axis()}, in.options());

  if (_AD_2nd_deriv)
  {
    // Set requires_grad to true if not already
    bool req_grad = in.tensor().requires_grad();
    in.tensor().requires_grad_();
    if (_AD_1st_deriv)
    {
      set_value(in, &out);
      // Loop over rows to retrieve the derivatives
      for (TorchSize i = 0; i < out.base_sizes()[0]; i++)
      {
        auto grad_outputs = BatchTensor::zeros_like(out);
        grad_outputs.index_put_({torch::indexing::Ellipsis, i}, 1.0);
        out.tensor().requires_grad_();
        auto jac_row = torch::autograd::grad({out},
                                             {in},
                                             {grad_outputs},
                                             /*retain_graph=*/true,
                                             /*create_graph=*/true,
                                             /*allow_unused=*/true)[0];
        if (jac_row.defined())
          dout_din.base_index_put({i, torch::indexing::Slice()}, jac_row);
      }
    }
    else
      set_value(in, &out, &dout_din);
    // Loop over rows to retrieve the second derivatives
    for (TorchSize i = 0; i < dout_din.base_sizes()[0]; i++)
      for (TorchSize j = 0; j < dout_din.base_sizes()[1]; j++)
      {
        auto grad_outputs = torch::zeros_like(dout_din);
        grad_outputs.index_put_({torch::indexing::Ellipsis, i, j}, 1.0);
        dout_din.tensor().requires_grad_();
        auto jac_row = torch::autograd::grad({dout_din},
                                             {in},
                                             {grad_outputs},
                                             /*retain_graph=*/true,
                                             /*create_graph=*/false,
                                             /*allow_unused=*/true)[0];
        if (jac_row.defined())
          d2out_din2.base_index_put({i, j, torch::indexing::Slice()}, jac_row);
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
Model::assemble(const BatchTensor & x, BatchTensor * r, BatchTensor * J) const
{
  auto in = LabeledVector::zeros(x.batch_sizes(), {&input()}, x.options());

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
