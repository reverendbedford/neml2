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

#include "neml2/drivers/TransientDriver.h"
#include "neml2/misc/math.h"

namespace fs = std::filesystem;

namespace neml2
{
OptionSet
TransientDriver::expected_options()
{
  OptionSet options = Driver::expected_options();

  options.set<std::string>("model");
  options.set("model").doc() = "The material model to be updated by the driver";

  options.set<CrossRef<torch::Tensor>>("times");
  options.set("times").doc() =
      "Time steps to perform the material update. The times tensor must have exactly 2 dimensions. "
      "The first dimension represents time steps, and the second dimension represents batches "
      "(i.e., how many material models to update simultaneously).";

  options.set<VariableName>("time") = VariableName("forces", "t");
  options.set("time").doc() = "Time";

  options.set<std::string>("predictor") = "PREVIOUS_STATE";
  options.set("predictor").doc() =
      "Predictor used to set the initial guess for each time step. Options are PREVIOUS_STATE, "
      "LINEAR_EXTRAPOLATION, CP_PREVIOUS_STATE, and CP_LINEAR_EXTRAPOLATION. The options prefixed "
      "with 'CP_' are specifically designed for crystal plasticity models.";

  options.set<Real>("cp_elastic_scale") = 1.0;
  options.set("cp_elastic_scale").doc() = "Elastic step scale factor used in the 'CP_' predictors";

  options.set<std::string>("save_as");
  options.set("save_as").doc() =
      "File path (absolute or relative to the working directory) to store the results";

  options.set<bool>("show_parameters") = false;
  options.set("show_parameters").doc() = "Whether to show model parameters at the beginning";

  options.set<bool>("show_input_axis") = false;
  options.set("show_input_axis").doc() = "Whether to show model input axis at the beginning";

  options.set<bool>("show_output_axis") = false;
  options.set("show_output_axis").doc() = "Whether to show model output axis at the beginning";

  options.set<std::string>("device") = "cpu";
  options.set("device").doc() =
      "Device on which to evaluate the material model. The string supplied must follow the "
      "following schema: (cpu|cuda)[:<device-index>] where cpu or cuda specifies the device type, "
      "and :<device-index> optionally specifies a device index. For example, device='cpu' sets the "
      "target compute device to be CPU, and device='cuda:1' sets the target compute device to be "
      "CUDA with device ID 1.";

  options.set<std::vector<VariableName>>("ic_scalar_names");
  options.set("ic_scalar_names").doc() = "Apply initial conditions to these Scalar variables";

  options.set<std::vector<CrossRef<Scalar>>>("ic_scalar_values");
  options.set("ic_scalar_values").doc() = "Initial condition values for the Scalar variables";

  options.set<std::vector<VariableName>>("ic_rot_names");
  options.set("ic_rot_names").doc() = "Apply initial conditions to these Rot variables";

  options.set<std::vector<CrossRef<Rot>>>("ic_rot_values");
  options.set("ic_rot_values").doc() = "Initial condition values for the Rot variables";

  options.set<std::vector<VariableName>>("ic_sr2_names");
  options.set("ic_sr2_names").doc() = "Apply initial conditions to these SR2 variables";

  options.set<std::vector<CrossRef<SR2>>>("ic_sr2_values");
  options.set("ic_sr2_values").doc() = "Initial condition values for the SR2 variables";

  return options;
}

TransientDriver::TransientDriver(const OptionSet & options)
  : Driver(options),
    _model(get_model(options.get<std::string>("model"), true, AssemblyMode::INPLACE)),
    _device(options.get<std::string>("device")),
    _time(options.get<CrossRef<torch::Tensor>>("times"), 2),
    _step_count(0),
    _time_name(options.get<VariableName>("time")),
    _nsteps(_time.batch_sizes()[0]),
    _nbatch(_time.batch_sizes()[1]),
    _in(_model.input_storage<LabeledVector>()),
    _out(_model.output_storage<LabeledVector>()),
    _predictor(options.get<std::string>("predictor")),
    _save_as(options.get<std::string>("save_as")),
    _show_params(options.get<bool>("show_parameters")),
    _show_input(options.get<bool>("show_input_axis")),
    _show_output(options.get<bool>("show_output_axis")),
    _result_in(_nsteps),
    _result_out(_nsteps),
    _ic_scalar_names(options.get<std::vector<VariableName>>("ic_scalar_names")),
    _ic_scalar_values(options.get<std::vector<CrossRef<Scalar>>>("ic_scalar_values")),
    _ic_rot_names(options.get<std::vector<VariableName>>("ic_rot_names")),
    _ic_rot_values(options.get<std::vector<CrossRef<Rot>>>("ic_rot_values")),
    _ic_sr2_names(options.get<std::vector<VariableName>>("ic_sr2_names")),
    _ic_sr2_values(options.get<std::vector<CrossRef<SR2>>>("ic_sr2_values")),
    _cp_elastic_scale(options.get<Real>("cp_elastic_scale"))
{
  _model.reinit({_nbatch}, 0, _device);

  _time = _time.to(_device);
}

void
TransientDriver::check_integrity() const
{
  Driver::check_integrity();

  auto errors = _model.preflight();
  // For now, let's throw one error at a time...
  if (!errors.empty())
    throw errors[0];

  neml_assert(_time.dim() == 2,
              "Input time should have dimension 2 but instead has dimension ",
              _time.dim());
}

bool
TransientDriver::run()
{
  // LCOV_EXCL_START
  if (_show_params)
    for (auto && [pname, pval] : _model.named_parameters())
      std::cout << pname << std::endl;

  if (_show_input)
    std::cout << _model.name() << "'s input axis:\n" << _model.input_axis() << std::endl;

  if (_show_output)
    std::cout << _model.name() << "'s output axis:\n" << _model.output_axis() << std::endl;
  // LCOV_EXCL_STOP

  auto status = solve();

  if (!save_as_path().empty())
    output();

  return status;
}

bool
TransientDriver::solve()
{
  for (_step_count = 0; _step_count < _nsteps; _step_count++)
  {
    if (_verbose)
      // LCOV_EXCL_START
      std::cout << "Step " << _step_count << std::endl;
    // LCOV_EXCL_STOP

    if (_step_count > 0)
      advance_step();
    update_forces();
    if (_step_count == 0)
    {
      store_input();
      apply_ic();
    }
    else
    {
      apply_predictor();
      store_input();
      solve_step();
    }
    store_output();

    if (_verbose)
      // LCOV_EXCL_START
      std::cout << std::endl;
    // LCOV_EXCL_STOP
  }

  return true;
}

void
TransientDriver::advance_step()
{
  if (_model.input_axis().has_subaxis("old_state") && _model.output_axis().has_subaxis("state"))
    _in.collect_(_out, "old_state", "state");

  if (_model.input_axis().has_subaxis("old_forces") && _model.input_axis().has_subaxis("forces"))
    _in.collect_(_in, "old_forces", "forces");
}

void
TransientDriver::update_forces()
{
  // Update current time
  _in.set_({_time_name}, _time.batch_index({_step_count}));
}

void
TransientDriver::apply_ic()
{
  set_IC<Scalar>(_ic_scalar_names, _ic_scalar_values);
  set_IC<Rot>(_ic_rot_names, _ic_rot_values);
  set_IC<SR2>(_ic_sr2_names, _ic_sr2_values);
}

void
TransientDriver::apply_predictor()
{
  std::string predictor = _predictor;
  bool cp = false;
  if (predictor.substr(0, 3) == "CP_")
  {
    predictor = predictor.substr(3);
    cp = true;
  }

  if (_model.input_axis().has_subaxis("state") && _model.input_axis().has_subaxis("old_state"))
  {
    if (predictor == "PREVIOUS_STATE")
      _in.collect_(_in, "state", "old_state");
    else if (predictor == "LINEAR_EXTRAPOLATION")
    {
      // Fall back to PREVIOUS_STATE predictor at the 1st time step
      if (_step_count == 1)
        _in.collect_(_in, "state", "old_state");
      // Otherwise linearly extrapolate in time
      else
      {
        auto t = _in.reinterpret<Scalar>({_time_name});
        auto t_n = _result_in[_step_count - 1].reinterpret<Scalar>({_time_name});
        auto t_nm1 = _result_in[_step_count - 2].reinterpret<Scalar>({_time_name});
        auto dt = t - t_n;
        auto dt_n = t_n - t_nm1;

        auto state_n = _result_out[_step_count - 1].get({"state"});
        auto state_nm1 = _result_out[_step_count - 2].get({"state"});
        auto state_np1 = state_n + (state_n - state_nm1) / dt_n * dt;
        _in.collect_(LabeledVector(state_np1, {&_model.output_axis().subaxis("state")}), "state");
      }
    }
    else
      throw NEMLException("Unrecognized predictor type: " + _predictor);
  }

  if (cp && (_step_count == 1))
  {
    SR2 D = _in.reinterpret<SR2>({VariableName{"forces", "deformation_rate"}});
    auto t = _in.reinterpret<Scalar>({_time_name});
    auto t_n = _result_in[_step_count - 1].reinterpret<Scalar>({_time_name});
    _in.set_({VariableName{"state", "elastic_strain"}}, D * (t - t_n) * _cp_elastic_scale);
  }
}

void
TransientDriver::solve_step()
{
  _model.value(_in.tensor());
}

void
TransientDriver::store_input()
{
  _result_in[_step_count] = _in.clone();
}

void
TransientDriver::store_output()
{
  _result_out[_step_count] = _out.clone();
}

std::string
TransientDriver::save_as_path() const
{
  return _save_as;
}

torch::nn::ModuleDict
TransientDriver::result() const
{
  // Dump input variables into a torch::nn::Module
  auto res_in = std::make_shared<torch::nn::Module>();
  for (auto var : _model.input_axis().variable_accessors(/*recursive=*/true))
  {
    std::vector<BatchTensor> step_vars(_nsteps);
    for (TorchSize i = 0; i < _nsteps; i++)
      step_vars[i] = _result_in[i].tensor().to(torch::kCPU).detach().clone();
    res_in->register_buffer(utils::stringify(var), math::stack(step_vars));
  }

  // Dump output variables into a torch::nn::Module
  auto res_out = std::make_shared<torch::nn::Module>();
  for (auto var : _model.output_axis().variable_accessors(/*recursive=*/true))
  {
    std::vector<BatchTensor> step_vars(_nsteps);
    for (TorchSize i = 0; i < _nsteps; i++)
      step_vars[i] = _result_out[i].tensor().to(torch::kCPU).detach().clone();
    res_out->register_buffer(utils::stringify(var), math::stack(step_vars));
  }

  // Combine input and output into a torch::nn::ModuleDict
  torch::nn::ModuleDict res;
  res->update({{"input", res_in}, {"output", res_out}});
  return res;
}

void
TransientDriver::output() const
{
  if (_verbose)
    // LCOV_EXCL_START
    std::cout << "Saving results..." << std::endl;
  // LCOV_EXCL_STOP

  auto cwd = fs::current_path();
  auto out = cwd / save_as_path();

  if (out.extension() == ".pt")
    output_pt(out);
  else
    // LCOV_EXCL_START
    neml_assert(false, "Unsupported output format: ", out.extension());
  // LCOV_EXCL_STOP

  if (_verbose)
    // LCOV_EXCL_START
    std::cout << "Results saved to " << save_as_path() << std::endl;
  // LCOV_EXCL_STOP
}

void
TransientDriver::output_pt(const std::filesystem::path & out) const
{
  torch::save(result(), out);
}
} // namespace neml2
