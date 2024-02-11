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
#include "neml2/models/ComposedModel.h"
#include "neml2/models/ImplicitUpdate.h"

namespace fs = std::filesystem;
using vecstr = std::vector<std::string>;

namespace neml2
{
OptionSet
TransientDriver::expected_options()
{
  OptionSet options = Driver::expected_options();
  options.set<std::string>("model");
  options.set<CrossRef<torch::Tensor>>("times");
  options.set<VariableName>("time") = vecstr{"forces", "t"};
  options.set<std::string>("predictor") = "PREVIOUS_STATE";
  options.set<Real>("cp_elastic_scale") = 1.0;
  options.set<std::string>("save_as");
  options.set<bool>("show_parameters") = false;
  options.set<bool>("show_input_axis") = false;
  options.set<bool>("show_output_axis") = false;
  options.set<std::string>("device") = "cpu";

  options.set<std::vector<VariableName>>("ic_scalar_names");
  options.set<std::vector<CrossRef<Scalar>>>("ic_scalar_values");
  options.set<std::vector<VariableName>>("ic_rot_names");
  options.set<std::vector<CrossRef<Rot>>>("ic_rot_values");
  options.set<std::vector<VariableName>>("ic_sr2_names");
  options.set<std::vector<CrossRef<SR2>>>("ic_sr2_values");

  return options;
}

TransientDriver::TransientDriver(const OptionSet & options)
  : Driver(options),
    _model(Factory::get_object<Model>("Models", options.get<std::string>("model"))),
    _device(options.get<std::string>("device")),
    _time(options.get<CrossRef<torch::Tensor>>("times"), 2),
    _step_count(0),
    _time_name(options.get<VariableName>("time")),
    _nsteps(_time.batch_sizes()[0]),
    _nbatch(_time.batch_sizes()[1]),
    _in(_model.input_storage()),
    _out(_model.output_storage()),
    _predictor(options.get<std::string>("predictor")),
    _save_as(options.get<std::string>("save_as")),
    _show_params(options.get<bool>("show_parameters")),
    _show_input(options.get<bool>("show_input_axis")),
    _show_output(options.get<bool>("show_output_axis")),
    _result_in(LabeledVector::zeros({_nsteps, _nbatch}, {&_model.input_axis()})),
    _result_out(LabeledVector::zeros({_nsteps, _nbatch}, {&_model.output_axis()})),
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
  _result_in = _result_in.to(_device);
  _result_out = _result_out.to(_device);
}

void
TransientDriver::check_integrity() const
{
  Driver::check_integrity();
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
  if (_in.axis(0).has_subaxis("old_state") && _out.axis(0).has_subaxis("state"))
    _in.slice("old_state").fill(_out.slice("state"));

  if (_in.axis(0).has_subaxis("old_forces") && _in.axis(0).has_subaxis("forces"))
    _in.slice("old_forces").fill(_in.slice("forces"));
}

void
TransientDriver::update_forces()
{
  auto current_time = _time.batch_index({_step_count});
  _in.set(current_time, _time_name);
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

  if (_in.axis(0).has_subaxis("state") && _in.axis(0).has_subaxis("old_state"))
  {
    if (predictor == "PREVIOUS_STATE")
      _in.slice("state").fill(_in.slice("old_state"));
    else if (predictor == "LINEAR_EXTRAPOLATION")
    {
      // Fall back to PREVIOUS_STATE predictor at the 1st time step
      if (_step_count == 1)
        _in.slice("state").fill(_in.slice("old_state"));
      // Otherwise linearly extrapolate in time
      else
      {
        auto t = _in.get<Scalar>(_time_name);
        auto t_n = _result_in.get<Scalar>(_time_name).batch_index({_step_count - 1});
        auto t_nm1 = _result_in.get<Scalar>(_time_name).batch_index({_step_count - 2});
        auto dt = t - t_n;
        auto dt_n = t_n - t_nm1;

        auto states = _result_out.slice("state");
        auto state_n = states.tensor().batch_index({_step_count - 1});
        auto state_nm1 = states.tensor().batch_index({_step_count - 2});
        LabeledVector state(state_n + (state_n - state_nm1) / dt_n * dt, states.axes());
        _in.slice("state").fill(state);
      }
    }
    else
      throw NEMLException("Unrecognized predictor type: " + _predictor);
  }

  if (cp && (_step_count == 1))
  {
    SR2 D = _in.get<SR2>(std::vector<std::string>{"forces", "deformation_rate"});
    auto t = _in.get<Scalar>(_time_name);
    auto t_n = _result_in.get<Scalar>(_time_name).batch_index({_step_count - 1});
    _in.set(D * (t - t_n) * _cp_elastic_scale, std::vector<std::string>{"state", "elastic_strain"});
  }
}

void
TransientDriver::solve_step()
{
  _model.value();
}

void
TransientDriver::store_input()
{
  _result_in.batch_index_put({_step_count}, _in);
}

void
TransientDriver::store_output()
{
  _result_out.batch_index_put({_step_count}, _out);
}

std::string
TransientDriver::save_as_path() const
{
  return _save_as;
}

torch::nn::ModuleDict
TransientDriver::result() const
{
  auto result_in_cpu = _result_in.to(torch::kCPU);
  auto result_out_cpu = _result_out.to(torch::kCPU);

  // Dump input variables into a Module
  auto res_in = std::make_shared<torch::nn::Module>();
  for (auto var : result_in_cpu.axis(0).variable_accessors(/*recursive=*/true))
    res_in->register_buffer(utils::stringify(var), result_in_cpu(var).clone());

  // Dump output variables into a Module
  auto res_out = std::make_shared<torch::nn::Module>();
  for (auto var : result_out_cpu.axis(0).variable_accessors(/*recursive=*/true))
    res_out->register_buffer(utils::stringify(var), result_out_cpu(var).clone());

  // Combine input and output
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
