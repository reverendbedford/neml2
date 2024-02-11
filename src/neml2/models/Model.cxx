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
  OptionSet options = Data::expected_options();
  options += NonlinearSystem::expected_options();
  options.set<std::vector<LabeledAxisAccessor>>("additional_outputs");
  options.set<bool>("use_AD_first_derivative") = false;
  options.set<bool>("use_AD_second_derivative") = false;
  options.set<int>("_extra_derivative_order") = 0;
  return options;
}

Model::Model(const OptionSet & options)
  : Data(options),
    ParameterStore(options, this),
    VariableStore(options, this),
    NonlinearSystem(options),
    _additional_outputs(options.get<std::vector<LabeledAxisAccessor>>("additional_outputs")),
    _AD_1st_deriv(options.get<bool>("use_AD_first_derivative")),
    _AD_2nd_deriv(options.get<bool>("use_AD_second_derivative")),
    _options(default_tensor_options()),
    _deriv_order(0),
    _extra_deriv_order(options.get<int>("_extra_derivative_order"))
{
  check_AD_limitation();
}

void
Model::setup()
{
  // Declare nonlinear parameters as input variable
  for (const auto & [name, param] : nl_params())
    declare_input_variable(param->name(), param->base_storage());

  // Setup input and output axes
  setup_layout();

  // Setup functional dependence for each output variable
  for (auto && [y_name, y_var] : output_views())
  {
    y_var.clear_args();
    for (auto && [x_name, x_var] : input_views())
      y_var.add_arg(x_var);
  }

  // Determine if this is an implicit system
  if (!input_axis().has_subaxis("state"))
    _implicit = false;
  else if (!output_axis().has_subaxis("residual"))
    _implicit = false;
  else if (input_axis().subaxis("state") != output_axis().subaxis("residual"))
    _implicit = false;
  else
    _implicit = true;
}

bool
Model::implicit() const
{
  return _implicit;
}

void
Model::reinit(TorchShapeRef batch_shape,
              int deriv_order,
              const torch::Device & device,
              const torch::Dtype & dtype)
{
  neml_assert(host() == this, "This method should only be called on the host model.");

  // Tensor options
  const auto options = default_tensor_options().device(device).dtype(dtype);

  // Cache batch shape, tensor options, and derivative order
  cache(batch_shape, options, deriv_order);

  // Send buffers and parameters
  send_buffers_to(options);
  send_parameters_to(options);

  // Reallocate variable storage
  allocate_variables(batch_shape, options);
  setup_input_views();
  setup_output_views(true, true, true);

  // Setup views for residual and Jacobian
  reinit_implicit_system(true, true, true);
}

void
Model::reinit(const BatchTensor & tensor, int deriv_order)
{
  reinit(tensor.batch_sizes(), deriv_order, tensor.device(), tensor.scalar_type());
}

void
Model::allocate_variables(TorchShapeRef batch_shape, const torch::TensorOptions & options)
{
  VariableStore::allocate_variables(batch_shape, options, _deriv_order);
  for (auto submodel : registered_models())
    submodel->allocate_variables(batch_shape, options);
}

void
Model::setup_input_views()
{
  VariableStore::setup_input_views();
  setup_submodel_input_views();
}

void
Model::setup_submodel_input_views()
{
  for (auto submodel : registered_models())
  {
    for (auto && [name, var] : submodel->input_views())
      var.setup_views(input_view(var.name()));

    submodel->setup_submodel_input_views();
  }
}

void
Model::setup_output_views(bool out, bool dout_din, bool d2out_din2)
{
  VariableStore::setup_output_views(
      out, dout_din && requires_grad(), d2out_din2 && requires_2nd_grad());
  setup_submodel_output_views(out, dout_din, d2out_din2);
}

void
Model::setup_submodel_output_views(bool out, bool dout_din, bool d2out_din2)
{
  for (auto submodel : registered_models())
  {
    for (auto && [name, var] : submodel->output_views())
      var.setup_views(
          out ? &submodel->output_storage() : nullptr,
          dout_din && submodel->requires_grad() ? &submodel->derivative_storage() : nullptr,
          d2out_din2 && submodel->requires_2nd_grad() ? &submodel->second_derivative_storage()
                                                      : nullptr);

    submodel->setup_submodel_output_views(out, dout_din, d2out_din2);
  }
}

void
Model::detach_and_zero(bool out, bool dout_din, bool d2out_din2)
{
  VariableStore::detach_and_zero(
      out, dout_din && requires_grad(), d2out_din2 && requires_2nd_grad());
  reinit_implicit_system(false, out, dout_din && requires_grad());

  for (auto submodel : registered_models())
    submodel->detach_and_zero(out, dout_din, d2out_din2);
}

void
Model::cache(TorchShapeRef batch_shape, const torch::TensorOptions & options, int deriv_order)
{
  _batch_sizes = batch_shape.vec();
  _options = options;
  _deriv_order = deriv_order + _extra_deriv_order;

  VariableStore::cache(batch_shape);

  for (auto submodel : registered_models())
    submodel->cache(batch_shape, options, _deriv_order);
}

void
Model::reinit_implicit_system(bool s, bool r, bool J)
{
  if (implicit())
  {
    if (s)
    {
      _ndof = host<Model>()->input_axis().storage_size("state");
      _solution = host<Model>()->input_storage()("state");
    }

    if (r)
      _residual = output_storage()("residual");

    if (J)
      _Jacobian = derivative_storage()("residual", "state");
  }

  for (auto submodel : registered_models())
    submodel->reinit_implicit_system(s, r, J);
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
Model::set_input(const LabeledVector & in)
{
  neml_assert_dbg(in.batch_sizes() == batch_sizes(),
                  "Incompatible batch shape between the model and the input. Did you call reinit? "
                  "The model has cached batch shape: ",
                  batch_sizes(),
                  ", but the input has batch shape: ",
                  in.batch_sizes());
  neml_assert_dbg(in.axis(0) == input_axis(),
                  "Incompatible input axis. The model has input axis: \n",
                  input_axis(),
                  "The input vector has axis: \n",
                  in.axis(0));

  input_storage().copy_(in);
}

LabeledVector
Model::get_output()
{
  return output_storage().clone();
}

LabeledMatrix
Model::get_doutput_dinput()
{
  return derivative_storage().clone();
}

LabeledTensor3D
Model::get_d2output_dinput2()
{
  return second_derivative_storage().clone();
}

LabeledVector
Model::value(const LabeledVector & in)
{
  set_input(in);
  value();
  return get_output();
}

std::tuple<LabeledVector, LabeledMatrix>
Model::value_and_dvalue(const LabeledVector & in)
{
  set_input(in);
  value_and_dvalue();
  return {get_output(), get_doutput_dinput()};
}

std::tuple<LabeledVector, LabeledMatrix, LabeledTensor3D>
Model::value_and_dvalue_and_d2value(const LabeledVector & in)
{
  set_input(in);
  value_and_dvalue_and_d2value();
  return {get_output(), get_doutput_dinput(), get_d2output_dinput2()};
}

void
Model::value()
{
  detach_and_zero(true, false, false);
  set_value(true, false, false);
}

void
Model::value_and_dvalue()
{
  neml_assert_dbg(requires_grad(),
                  "value_and_dvalue() is called but derivative storage hasn't been allocated.");

  if (!_AD_1st_deriv)
  {
    detach_and_zero(true, true, false);
    set_value(true, true, false);
  }
  else
  {
    input_storage().tensor().requires_grad_();
    detach_and_zero(true, false, false);
    set_value(true, false, false);
    extract_derivatives(/*retain_graph=*/true, /*create_graph=*/false, /*allow_unused=*/true);
    input_storage().tensor().requires_grad_(false);
  }
}

void
Model::value_and_dvalue_and_d2value()
{
  neml_assert_dbg(requires_2nd_grad(),
                  "value_and_dvalue_and_d2value() is called but second derivative storage hasn't "
                  "been allocated.");

  if (!_AD_2nd_deriv)
  {
    detach_and_zero(true, true, true);
    set_value(true, true, true);
  }
  else
  {
    input_storage().tensor().requires_grad_();
    if (!_AD_1st_deriv)
    {
      detach_and_zero(true, true, false);
      set_value(true, true, false);
    }
    else
    {
      detach_and_zero(true, false, false);
      set_value(true, false, false);
      extract_derivatives(/*retain_graph=*/true, /*create_graph=*/true, /*allow_unused=*/true);
    }
    extract_second_derivatives(
        /*retain_graph=*/true, /*create_graph=*/false, /*allow_unused=*/true);
    input_storage().tensor().requires_grad_(false);
  }
}

const std::set<LabeledAxisAccessor>
Model::consumed_items() const
{
  return input_axis().variable_accessors(true);
}

const std::set<LabeledAxisAccessor>
Model::provided_items() const
{
  return output_axis().variable_accessors(true);
}

void
Model::assemble(bool residual, bool Jacobian)
{
  if (residual && !Jacobian)
    value();
  else if (Jacobian)
    value_and_dvalue();
}

void
Model::extract_derivatives(bool retain_graph, bool create_graph, bool allow_unused)
{
  // Loop over rows to retrieve the derivatives
  if (output_storage().tensor().requires_grad())
    for (TorchSize i = 0; i < output_storage().base_sizes()[0]; i++)
    {
      auto grad_outputs = BatchTensor::zeros_like(output_storage());
      grad_outputs.index_put_({torch::indexing::Ellipsis, i}, 1.0);
      auto jac_row = torch::autograd::grad({output_storage()},
                                           {input_storage()},
                                           {grad_outputs},
                                           retain_graph,
                                           create_graph,
                                           allow_unused)[0];
      if (jac_row.defined())
        derivative_storage().base_index_put({i, torch::indexing::Slice()}, jac_row);
    }
}

void
Model::extract_second_derivatives(bool retain_graph, bool create_graph, bool allow_unused)
{
  // Loop over rows to retrieve the second derivatives
  if (derivative_storage().tensor().requires_grad())
    for (TorchSize i = 0; i < derivative_storage().base_sizes()[0]; i++)
      for (TorchSize j = 0; j < derivative_storage().base_sizes()[1]; j++)
      {
        auto grad_outputs = torch::zeros_like(derivative_storage());
        grad_outputs.index_put_({torch::indexing::Ellipsis, i, j}, 1.0);
        auto jac_row = torch::autograd::grad({derivative_storage()},
                                             {input_storage()},
                                             {grad_outputs},
                                             retain_graph,
                                             create_graph,
                                             allow_unused)[0];
        second_derivative_storage().base_index_put({i, j, torch::indexing::Slice()}, jac_row);
      }
}
} // namespace neml2
