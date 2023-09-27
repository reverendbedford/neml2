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

namespace neml2
{
OptionSet
TransientDriver::expected_options()
{
  OptionSet options = Driver::expected_options();
  options.set<std::string>("model");
  options.set<CrossRef<torch::Tensor>>("times");
  options.set<LabeledAxisAccessor>("time") = LabeledAxisAccessor{{"forces", "t"}};
  options.set<std::string>("predictor") = "PREVIOUS_STATE";
  options.set<std::string>("save_as");
  options.set<bool>("show_parameters") = false;
  options.set<std::string>("device") = "cpu";
  return options;
}

TransientDriver::TransientDriver(const OptionSet & options)
  : Driver(options),
    _model(Factory::get_object<Model>("Models", options.get<std::string>("model"))),
    _device(options.get<std::string>("device")),
    _time(options.get<CrossRef<torch::Tensor>>("times"), 2),
    _step_count(0),
    _time_name(options.get<LabeledAxisAccessor>("time")),
    _nsteps(_time.batch_sizes()[0]),
    _nbatch(_time.batch_sizes()[1]),
    _in(LabeledVector::zeros(_nbatch, {&_model.input()})),
    _out(LabeledVector::zeros(_nbatch, {&_model.output()})),
    _predictor(options.get<std::string>("predictor")),
    _save_as(options.get<std::string>("save_as")),
    _show_params(options.get<bool>("show_parameters")),
    _result_in(LabeledVector::zeros({_nsteps, _nbatch}, {&_model.input()})),
    _result_out(LabeledVector::zeros({_nsteps, _nbatch}, {&_model.output()}))
{
  _model.to(_device);
  _in = _in.to(_device);
  _out = _out.to(_device);
  _time = _time.to(_device);
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
  if (_show_params)
    // LCOV_EXCL_START
    for (auto & item : _model.named_parameters(true))
      std::cout << item.key() << std::endl;
  // LCOV_EXCL_STOP

  auto status = solve();

  if (!save_as_path().empty())
    output();

  return status;
}

bool
TransientDriver::solve()
{
  // We don't need parameter gradients
  torch::NoGradGuard no_grad_guard;

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
      apply_ic();
    else
    {
      if (_model.implicit())
        apply_predictor();
      solve_step();
    }
    store_step();

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
  LabeledVector(_in.slice("old_state")).fill(_out.slice("state"));
  LabeledVector(_in.slice("old_forces")).fill(_in.slice("forces"));
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
}

void
TransientDriver::apply_predictor()
{
  if (_predictor == "PREVIOUS_STATE")
    _in.slice("state").fill(_in.slice("old_state"));
  else if (_predictor == "LINEAR_EXTRAPOLATION")
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

void
TransientDriver::solve_step()
{
  _out = _model.value(_in);
}

void
TransientDriver::store_step()
{
  _result_in.batch_index_put({_step_count}, _in);
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
  // Dump input variables into a Module
  auto res_in = std::make_shared<torch::nn::Module>();
  for (auto var : _result_in.axis(0).variable_accessors(/*recursive=*/true))
    res_in->register_buffer(utils::stringify(var), _result_in(var).clone());

  // Dump output variables into a Module
  auto res_out = std::make_shared<torch::nn::Module>();
  for (auto var : _result_out.axis(0).variable_accessors(/*recursive=*/true))
    res_out->register_buffer(utils::stringify(var), _result_out(var).clone());

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
