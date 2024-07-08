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
#include "neml2/misc/utils.h"
#include "neml2/misc/math.h"

namespace neml2
{
Model::Stage Model::stage = UPDATING;

OptionSet
Model::expected_options()
{
  OptionSet options = Data::expected_options();
  options += VariableStore::expected_options();
  options += NonlinearSystem::expected_options();
  NonlinearSystem::disable_automatic_scaling(options);

  options.section() = "Models";

  options.set<bool>("_use_AD_first_derivative") = false;
  options.set<bool>("_use_AD_second_derivative") = false;
  options.set<int>("_extra_derivative_order") = 0;
  options.set<bool>("_nonlinear_system") = false;

  options.set("_extra_derivative_order").suppressed() = true;
  options.set("_nonlinear_system").suppressed() = true;
  options.set("_use_AD_first_derivative").suppressed() = true;
  options.set("_use_AD_second_derivative").suppressed() = true;

  return options;
}

Model::Model(const OptionSet & options)
  : Data(options),
    ParameterStore(options, this),
    VariableStore(options, this),
    NonlinearSystem(options),
    _AD_1st_deriv(options.get<bool>("_use_AD_first_derivative")),
    _AD_2nd_deriv(options.get<bool>("_use_AD_second_derivative")),
    _options(default_tensor_options()),
    _deriv_order(-1),
    _extra_deriv_order(options.get<int>("_extra_derivative_order")),
    _nonlinear_system(options.get<bool>("_nonlinear_system"))
{
  check_AD_limitation();
}

std::vector<Diagnosis>
Model::preflight() const
{
  neml_assert(host() == this, "This method should only be called on the host model.");

  std::vector<Diagnosis> errors;

  // Check for statefulness
  if (input_axis().has_subaxis("old_state"))
  {
    if (!output_axis().has_subaxis("state"))
      errors.push_back(
          make_diagnosis(name(),
                         ": input axis has sub-axis 'old_state', but output axis does not "
                         "have sub-axis 'state'."));
    else
    {
      auto s_vars = output_axis().subaxis("state").variable_accessors(/*recursive=*/true);
      for (auto var : input_axis().subaxis("old_state").variable_accessors(/*recursive=*/true))
        if (!s_vars.count(var))
          errors.push_back(make_diagnosis(name(),
                                          ": input axis has old state named ",
                                          var,
                                          ", but it doesn't exist on the output axis."));
    }
  }

  return errors;
}

void
Model::setup()
{
  // Setup input and output axes
  setup_layout();
}

void
Model::reinit(TorchShapeRef batch_shape,
              int deriv_order,
              const torch::Device & device,
              const torch::Dtype & dtype)
{
  neml_assert(host() == this, "This method should only be called on the host model.");

  // Cache configuration
  cache(batch_shape, deriv_order, device, dtype);

  // Sync buffers and parameters
  send_buffers_to(options());
  send_parameters_to(options());

  // Allocate variable storage
  allocate_variables();

  // Setup variable views
  setup_input_views();
  setup_output_views();
  setup_nonlinear_system();
}

void
Model::reinit(const BatchTensor & tensor, int deriv_order)
{
  reinit(tensor.batch_sizes(), deriv_order, tensor.device(), tensor.scalar_type());
}

void
Model::allocate_variables()
{
  VariableStore::allocate_variables(batch_sizes(),
                                    options(),
                                    /*in=*/true,
                                    /*out=*/true,
                                    /*dout_din=*/requires_grad(),
                                    /*d2out_din2=*/requires_2nd_grad());

  for (auto submodel : registered_models())
    submodel->allocate_variables();
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
    submodel->setup_input_views();
}

void
Model::setup_output_views()
{
  VariableStore::setup_output_views(true, requires_grad(), requires_2nd_grad());
  setup_submodel_output_views();
}

void
Model::setup_submodel_output_views()
{
  for (auto submodel : registered_models())
    submodel->setup_output_views();
}

void
Model::setup_nonlinear_system()
{
  if (is_nonlinear_system())
  {
    _ndof = output_axis().storage_size("residual");
    _solution = BatchTensor::empty(batch_sizes(), _ndof, options());
    _residual_view = &output_storage().view_raw({"residual"});
    if (requires_grad())
      _Jacobian_view = &derivative_storage().view_raw({"residual", "state"});
  }

  for (auto submodel : registered_models())
    submodel->setup_nonlinear_system();
}

void
Model::detach_and_zero(bool out, bool dout_din, bool d2out_din2)
{
  VariableStore::detach_and_zero(out, dout_din, d2out_din2);

  for (auto submodel : registered_models())
    submodel->detach_and_zero(out, dout_din, d2out_din2);
}

void
Model::set_solution(const BatchTensor & x)
{
  NonlinearSystem::set_solution(x);

  // Also update the model input variables
  host<VariableStore>()->input_storage().set_({"state"}, x);
}

void
Model::cache(TorchShapeRef batch_shape,
             int deriv_order,
             const torch::Device & device,
             const torch::Dtype & dtype)
{
  _batch_sizes = batch_shape.empty() ? TorchShape{1} : batch_shape.vec();
  VariableStore::cache(_batch_sizes);

  _deriv_order = std::max(deriv_order + _extra_deriv_order, _deriv_order);

  _options = default_tensor_options().device(device).dtype(dtype);

  for (auto submodel : registered_models())
    submodel->cache(batch_shape, _deriv_order, device, dtype);
}

void
Model::check_AD_limitation() const
{
  if (_AD_1st_deriv && !_AD_2nd_deriv)
    throw NEMLException("AD derivative is requested, but AD second derivative is not requested.");
}

void
Model::input_requires_grad_(bool req)
{
  for (auto && [name, var] : input_views())
    var.requires_grad_(req);
}

void
Model::use_AD_derivatives(bool first, bool second)
{
  _AD_1st_deriv = first;
  _AD_2nd_deriv = second;
  check_AD_limitation();
}

void
Model::set_input(const BatchTensor & in)
{
  neml_assert(utils::sizes_broadcastable(in.batch_sizes(), batch_sizes()));

  neml_assert(!in.requires_grad(), "Inplace assembly mode does not support AD");
  input_storage().copy_(in.batch_expand(batch_sizes()));
}

BatchTensor
Model::get_output()
{
  return output_storage().assemble();
}

BatchTensor
Model::get_doutput_dinput()
{
  return derivative_storage().assemble();
}

BatchTensor
Model::get_d2output_dinput2()
{
  return second_derivative_storage().assemble();
}

BatchTensor
Model::value(const BatchTensor & in)
{
  set_input(in);
  detach_and_zero(true, false, false);
  value();
  return get_output();
}

std::tuple<BatchTensor, BatchTensor>
Model::value_and_dvalue(const BatchTensor & in)
{
  set_input(in);
  detach_and_zero(true, true, false);
  value_and_dvalue();
  return {get_output(), get_doutput_dinput()};
}

std::tuple<BatchTensor, BatchTensor, BatchTensor>
Model::value_and_dvalue_and_d2value(const BatchTensor & in)
{
  set_input(in);
  detach_and_zero(true, true, true);
  value_and_dvalue_and_d2value();
  return {get_output(), get_doutput_dinput(), get_d2output_dinput2()};
}

void
Model::value()
{
  set_value(true, false, false);
}

void
Model::value_and_dvalue()
{
  neml_assert_dbg(requires_grad(),
                  "value_and_dvalue() is called but derivative storage hasn't been allocated.");

  if (!_AD_1st_deriv)
    set_value(true, true, false);
  else
  {
    input_requires_grad_();
    set_value(true, false, false);
    extract_derivatives(/*retain_graph=*/true, /*create_graph=*/true, /*allow_unused=*/true);
  }
}

void
Model::value_and_dvalue_and_d2value()
{
  neml_assert_dbg(requires_2nd_grad(),
                  "value_and_dvalue_and_d2value() is called but second derivative storage hasn't "
                  "been allocated.");

  if (!_AD_2nd_deriv)
    set_value(true, true, true);
  else
  {
    input_requires_grad_();

    if (!_AD_1st_deriv)
      set_value(true, true, false);
    else
    {
      set_value(true, false, false);
      extract_derivatives(/*retain_graph=*/true, /*create_graph=*/true, /*allow_unused=*/true);
    }

    extract_second_derivatives(
        /*retain_graph=*/true, /*create_graph=*/true, /*allow_unused=*/true);
  }
}

Model *
Model::registered_model(const std::string & name) const
{
  for (auto submodel : _registered_models)
    if (submodel->name() == name)
      return submodel;

  throw NEMLException("There is no registered model named '" + name + "' in '" + this->name() +
                      "'");
}

const std::set<VariableName>
Model::consumed_items() const
{
  return input_axis().variable_accessors(true);
}

const std::set<VariableName>
Model::provided_items() const
{
  return output_axis().variable_accessors(true);
}

void
Model::assemble(bool residual, bool Jacobian)
{
  if (residual && !Jacobian)
  {
    detach_and_zero(true, false, false);
    value();
    _residual = _residual_view->clone();
  }
  else if (Jacobian)
  {
    detach_and_zero(true, true, false);
    value_and_dvalue();
    _residual = _residual_view->clone();
    _Jacobian = _Jacobian_view->clone();
  }
}

void
Model::extract_derivatives(bool retain_graph, bool create_graph, bool allow_unused)
{
  neml_assert_dbg(assembly_mode() == AssemblyMode::CONCATENATION,
                  "This method only works with concatenation assembly mode");

  // Loop over rows to retrieve the derivatives
  for (auto && [yname, yvar] : output_views())
  {
    auto yten = yvar.raw_value();

    if (!yten.requires_grad())
      continue;

    for (auto && [xname, xvar] : input_views())
    {
      auto xten = xvar.raw_value();
      std::vector<BatchTensor> dy_dx(yten.base_sizes()[0]);
      for (TorchSize i = 0; i < yten.base_sizes()[0]; i++)
      {
        auto G = torch::zeros_like(yten, torch::TensorOptions().requires_grad(false));
        G.index_put_({torch::indexing::Ellipsis, i}, 1.0);
        dy_dx[i] = BatchTensor(
            torch::autograd::grad({yten}, {xten}, {G}, retain_graph, create_graph, allow_unused)[0],
            yten.batch_dim());
      }
      derivative_storage().set_({yname, xname}, math::stack(dy_dx, -2));
    }
  }
}

void
Model::extract_second_derivatives(bool retain_graph, bool create_graph, bool allow_unused)
{
  neml_assert_dbg(assembly_mode() == AssemblyMode::CONCATENATION,
                  "This method only works with concatenation assembly mode");

  // Loop over rows to retrieve the second derivatives
  for (auto && [yname, yvar] : output_views())
    for (auto && [x1name, x1var] : input_views())
    {
      const auto dy_dx = derivative_storage().get({yname, x1name});

      if (!dy_dx.requires_grad())
        continue;

      for (auto && [x2name, x2var] : input_views())
      {
        auto xten = x2var.raw_value();
        std::vector<BatchTensor> d2y_dx2_row(dy_dx.base_sizes()[0]);
        for (TorchSize i = 0; i < dy_dx.base_sizes()[0]; i++)
        {
          std::vector<BatchTensor> d2y_dx2_col(dy_dx.base_sizes()[1]);
          for (TorchSize j = 0; j < dy_dx.base_sizes()[1]; j++)
          {
            auto G = torch::zeros_like(dy_dx, torch::TensorOptions().requires_grad(false));
            G.index_put_({torch::indexing::Ellipsis, i, j}, 1.0);
            d2y_dx2_col[j] =
                BatchTensor(torch::autograd::grad(
                                {dy_dx}, {xten}, {G}, retain_graph, create_graph, allow_unused)[0],
                            dy_dx.batch_dim());
          }
          d2y_dx2_row[i] = math::stack(d2y_dx2_col, -2);
        }
        second_derivative_storage().set_({yname, x1name, x2name}, math::stack(d2y_dx2_row, -3));
      }
    }
}
} // namespace neml2
