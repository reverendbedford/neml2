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

#include "neml2/models/NewModel.h"

namespace neml2
{
NewModel::Stage NewModel::stage = UPDATING;

OptionSet
NewModel::expected_options()
{
  OptionSet options = Data::expected_options();
  options += NonlinearSystem::expected_options();
  options.set<std::vector<LabeledAxisAccessor>>("additional_outputs");
  options.set<bool>("use_AD_first_derivative") = false;
  options.set<bool>("use_AD_second_derivative") = false;
  return options;
}

NewModel::NewModel(const OptionSet & options)
  : Data(options),
    ParameterStore(options, this),
    VariableStore(options, this),
    NonlinearSystem(options),
    _additional_outputs(options.get<std::vector<LabeledAxisAccessor>>("additional_outputs")),
    _AD_1st_deriv(options.get<bool>("use_AD_first_derivative")),
    _AD_2nd_deriv(options.get<bool>("use_AD_second_derivative")),
    _options(default_tensor_options())
{
  check_AD_limitation();
}

void
NewModel::setup()
{
  setup_layout();

  for (auto && [y_name, y_var] : output_views())
  {
    y_var.clear_args();
    for (auto && [x_name, x_var] : input_views())
      y_var.add_arg(x_var);
  }
}

bool
NewModel::implicit() const
{
  if (!host<NewModel>()->input_axis().has_subaxis("state"))
    return false;

  if (!output_axis().has_subaxis("residual"))
    return false;

  if (host<NewModel>()->input_axis().subaxis("state") != output_axis().subaxis("residual"))
    return false;

  return true;
}

void
NewModel::reinit(TorchShapeRef batch_shape, const torch::TensorOptions & options)
{
  neml_assert(host() == this, "This method should only be called on the host model.");

  // Cache batch shape and tensor options
  cache(batch_shape);
  cache(options);

  // Send buffers and parameters
  send_buffers_to(options);
  send_parameters_to(options);

  // Reallocate variable storage
  allocate_variables(batch_shape, options);
  setup_input_views();
  setup_output_views(true, true, true);

  // Setup views for residual and Jacobian
  reinit_implicit_system();
}

void
NewModel::reinit(const BatchTensor & tensor)
{
  reinit(tensor.batch_sizes(), tensor.options());
}

void
NewModel::allocate_variables(TorchShapeRef batch_shape, const torch::TensorOptions & options)
{
  VariableStore::allocate_variables(batch_shape, options);
  for (auto submodel : registered_models())
    submodel->allocate_variables(batch_shape, options);
}

void
NewModel::setup_input_views()
{
  VariableStore::setup_input_views();
  setup_submodel_input_views();
}

void
NewModel::setup_submodel_input_views()
{
  for (auto submodel : registered_models())
  {
    for (auto && [name, var] : submodel->input_views())
      var.setup_views(input_view(var.name()));

    submodel->setup_submodel_input_views();
  }
}

void
NewModel::setup_output_views(bool out, bool dout_din, bool d2out_din2)
{
  VariableStore::setup_output_views(out, dout_din, d2out_din2);
  setup_submodel_output_views(out, dout_din, d2out_din2);
}

void
NewModel::setup_submodel_output_views(bool out, bool dout_din, bool d2out_din2)
{
  for (auto submodel : registered_models())
  {
    for (auto && [name, var] : submodel->output_views())
      var.setup_views(out ? &submodel->output_storage() : nullptr,
                      dout_din ? &submodel->derivative_storage() : nullptr,
                      d2out_din2 ? &submodel->second_derivative_storage() : nullptr);

    submodel->setup_submodel_output_views(out, dout_din, d2out_din2);
  }
}

void
NewModel::detach_and_zero(bool out, bool dout_din, bool d2out_din2)
{
  VariableStore::detach_and_zero(out, dout_din, d2out_din2);
  for (auto submodel : registered_models())
    submodel->detach_and_zero(out, dout_din, d2out_din2);
}

void
NewModel::cache(TorchShapeRef batch_shape)
{
  _batch_sizes = batch_shape.vec();
  VariableStore::cache(batch_shape);

  for (auto submodel : registered_models())
    submodel->cache(batch_shape);
}

void
NewModel::cache(const torch::TensorOptions & options)
{
  _options = options;
  for (auto submodel : registered_models())
    submodel->cache(options);
}

void
NewModel::reinit_implicit_system()
{
  if (implicit())
  {
    _ndof = host<NewModel>()->input_axis().storage_size("state");
    _solution = host<NewModel>()->input_storage()("state");
    _residual = output_storage()("residual");
    _Jacobian = derivative_storage()("residual", "state");
  }

  for (auto submodel : registered_models())
    submodel->reinit_implicit_system();
}

void
NewModel::check_AD_limitation() const
{
  if (_AD_1st_deriv && !_AD_2nd_deriv)
    throw NEMLException("AD derivative is requested, but AD second derivative is not requested.");
}

void
NewModel::use_AD_derivatives(bool first, bool second)
{
  _AD_1st_deriv = first;
  _AD_2nd_deriv = second;
  check_AD_limitation();
}

void
NewModel::set_input(const LabeledVector & in)
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
NewModel::get_output()
{
  return output_storage().clone();
}

LabeledMatrix
NewModel::get_doutput_dinput()
{
  return derivative_storage().clone();
}

LabeledTensor3D
NewModel::get_d2output_dinput2()
{
  return second_derivative_storage().clone();
}

LabeledVector
NewModel::value(const LabeledVector & in)
{
  set_input(in);
  value();
  return get_output();
}

std::tuple<LabeledVector, LabeledMatrix>
NewModel::value_and_dvalue(const LabeledVector & in)
{
  set_input(in);
  value_and_dvalue();
  return {get_output(), get_doutput_dinput()};
}

std::tuple<LabeledVector, LabeledMatrix, LabeledTensor3D>
NewModel::value_and_dvalue_and_d2value(const LabeledVector & in)
{
  set_input(in);
  value_and_dvalue_and_d2value();
  return {get_output(), get_doutput_dinput(), get_d2output_dinput2()};
}

void
NewModel::value()
{
  detach_and_zero(true, false, false);
  set_value(true, false, false);
}

void
NewModel::value_and_dvalue()
{
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
NewModel::value_and_dvalue_and_d2value()
{
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
NewModel::consumed_items() const
{
  return input_axis().variable_accessors(true);
}

const std::set<LabeledAxisAccessor>
NewModel::provided_items() const
{
  return output_axis().variable_accessors(true);
}

void
NewModel::assemble(bool residual, bool Jacobian)
{
  // Let's try to be as efficient as possible by considering all the cases!
  if (residual && !Jacobian)
    value();
  else if (Jacobian)
    value_and_dvalue();
}

void
NewModel::extract_derivatives(bool retain_graph, bool create_graph, bool allow_unused)
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
NewModel::extract_second_derivatives(bool retain_graph, bool create_graph, bool allow_unused)
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
