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

#include "neml2/drivers/solid_mechanics/SolidMechanicsDriver.h"

namespace neml2
{
register_NEML2_object(SolidMechanicsDriver);

ParameterSet
SolidMechanicsDriver::expected_params()
{
  ParameterSet params = TransientDriver::expected_params();
  params.set<std::string>("control") = "STRAIN";
  params.set<LabeledAxisAccessor>("total_strain") = LabeledAxisAccessor{{"forces", "E"}};
  params.set<LabeledAxisAccessor>("cauchy_stress") = LabeledAxisAccessor{{"forces", "S"}};
  params.set<CrossRef<torch::Tensor>>("prescribed_strains");
  params.set<CrossRef<torch::Tensor>>("prescribed_stresses");
  return params;
}

SolidMechanicsDriver::SolidMechanicsDriver(const ParameterSet & params)
  : TransientDriver(params),
    _control(params.get<std::string>("control"))
{
  if (_control == "STRAIN")
  {
    _driving_force = params.get<CrossRef<torch::Tensor>>("prescribed_strains");
    _driving_force_name = params.get<LabeledAxisAccessor>("total_strain");
  }
  else if (_control == "STRESS")
  {
    _driving_force = params.get<CrossRef<torch::Tensor>>("prescribed_stresses");
    _driving_force_name = params.get<LabeledAxisAccessor>("cauchy_stress");
  }
  else
    // LCOV_EXCL_START
    neml_assert(false, "Unsupported control type.");
  // LCOV_EXCL_STOP

  check_integrity();
}

void
SolidMechanicsDriver::check_integrity() const
{
  TransientDriver::check_integrity();
  neml_assert(_driving_force.dim() == 3,
              "Input strain/stress should have dimension 3 but instead has dimension",
              _driving_force.dim());
  neml_assert(_time.sizes()[0] == _driving_force.sizes()[0],
              "Input strain/stress and time should have the same number of time steps. The input "
              "time has ",
              _time.sizes()[0],
              " time steps, while the input strain/stress has ",
              _driving_force.sizes()[0],
              " time steps");
  neml_assert(_time.sizes()[1] == _driving_force.sizes()[1],
              "Input strain/stress and time should have the same batch size. The input time has a "
              "batch size of ",
              _time.sizes()[1],
              " while the input strain/stress has a batch size of ",
              _driving_force.sizes()[1]);
  neml_assert(_driving_force.sizes()[2] == 6,
              "Input strain/stress should have final dimension 6 but instead has final dimension ",
              _driving_force.sizes()[2]);
}

bool
SolidMechanicsDriver::solve()
{
  // We don't need parameter gradients
  // torch::NoGradGuard no_grad_guard;

  for (_step_count = 0; _step_count < _nsteps; _step_count++)
    solve_step(_step_count == 0);

  return true;
}

void
SolidMechanicsDriver::solve_step(bool init)
{
  if (_verbose)
    // LCOV_EXCL_START
    std::cout << "Step " << _step_count << std::endl;
  // LCOV_EXCL_STOP

  // Advance the step
  if (!init)
    _model.advance_step();
  auto current_time = Scalar(_time.index({_step_count}).unsqueeze(-1));
  auto current_driving_force = SymR2(_driving_force.index({_step_count}));

  // Prepare input
  _in.set(current_driving_force, _driving_force_name);
  _in.set(current_time, _time_name);

  // Solve the step
  if (!init)
    _out = _model.value(_in);

  // Propagate the forces and state in time
  // current --> old
  // Initialize the old state it if necessary
  // if (init)
  //   _model.init_state(in);
  if (!init)
  {
    LabeledVector(_in.slice("old_state")).fill(_out.slice("state"));
    LabeledVector(_in.slice("old_forces")).fill(_in.slice("forces"));
  }

  // Store the results
  _result->push_back({{"input", _in}, {"output", _out}});

  if (_verbose)
    // LCOV_EXCL_START
    std::cout << std::endl;
  // LCOV_EXCL_STOP
}
}
