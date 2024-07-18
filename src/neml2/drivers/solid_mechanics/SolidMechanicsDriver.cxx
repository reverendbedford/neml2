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

OptionSet
SolidMechanicsDriver::expected_options()
{
  OptionSet options = TransientDriver::expected_options();
  options.doc() =
      "Driver for small deformation solid mechanics material model with optional thermal coupling.";

  options.set<std::string>("control") = "STRAIN";
  options.set("control").doc() = "External control of the material update. Options are STRAIN and "
                                 "STRESS, for strain control and stress control, respectively.";

  options.set<VariableName>("total_strain") = VariableName("forces", "E");
  options.set("total_strain").doc() = "Total strain";

  options.set<VariableName>("cauchy_stress") = VariableName("forces", "S");
  options.set("cauchy_stress").doc() = "Cauchy stress";

  options.set<VariableName>("temperature") = VariableName("forces", "T");
  options.set("temperature").doc() = "Name of temperature";

  options.set<VariableName>("fixed_values") = VariableName("forces", "fixed_values");
  options.set("fixed_values").doc() = "Name of fixed values (when control = MIXED)";

  options.set<CrossRef<torch::Tensor>>("prescribed_strains");
  options.set("prescribed_strains").doc() = "Prescribed strain (when control = STRAIN)";

  options.set<CrossRef<torch::Tensor>>("prescribed_stresses");
  options.set("prescribed_stresses").doc() = "Prescribed stress (when control = STRESS)";

  options.set<CrossRef<torch::Tensor>>("prescribed_temperatures");
  options.set("prescribed_temperatures").doc() =
      "Actual prescibed temperature values, when providing temperatures to the model";

  options.set<CrossRef<torch::Tensor>>("prescribed_mixed_conditions");
  options.set("prescribed_mixed_conditions").doc() =
      "The fixed, controlled values provided as user input for the mixed control case.  Where the "
      "control signal is 0 these are strain values, where it is 1 these are stress values";

  options.set<VariableName>("control_name") = VariableName("forces", "control");
  options.set("control_name").doc() = "The name of the control signal on the input axis";

  options.set<CrossRef<torch::Tensor>>("prescribed_control");
  options.set("prescribed_control").doc() = "The actual values of the control signal.  0 implies "
                                            "strain control, 1 implies stress control";

  return options;
}

SolidMechanicsDriver::SolidMechanicsDriver(const OptionSet & options)
  : TransientDriver(options),
    _control(options.get<std::string>("control")),
    _control_name(options.get<VariableName>("control_name")),
    _temperature_name(options.get<VariableName>("temperature")),
    _temperature_prescribed(
        !options.get<CrossRef<torch::Tensor>>("prescribed_temperatures").raw().empty()),
    _temperature(_temperature_prescribed
                     ? Scalar(options.get<CrossRef<torch::Tensor>>("prescribed_temperatures"), 2)
                     : Scalar()),
    _control_signal(_control == "MIXED"
                        ? SR2(options.get<CrossRef<torch::Tensor>>("prescribed_control"), 2)
                        : SR2())

{
  if (_control == "STRAIN")
  {
    _driving_force = SR2(options.get<CrossRef<torch::Tensor>>("prescribed_strains"), 2);
    _driving_force_name = options.get<VariableName>("total_strain");
  }
  else if (_control == "STRESS")
  {
    _driving_force = SR2(options.get<CrossRef<torch::Tensor>>("prescribed_stresses"), 2);
    _driving_force_name = options.get<VariableName>("cauchy_stress");
  }
  else if (_control == "MIXED")
  {
    _driving_force = SR2(options.get<CrossRef<torch::Tensor>>("prescribed_mixed_conditions"), 2);
    _driving_force_name = options.get<VariableName>("fixed_values");
  }
  else
    // LCOV_EXCL_START
    throw NEMLException("Unsupported control type.");
  // LCOV_EXCL_STOP

  _driving_force = _driving_force.to(_device);

  if (_temperature_prescribed)
    _temperature = _temperature.to(_device);

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

  if (_temperature_prescribed)
  {
    neml_assert(_temperature.batch_dim() == 2,
                "Input temperature should have 2 batch dimensions but instead has batch dimension",
                _temperature.batch_dim());
    neml_assert(_time.sizes()[0] == _temperature.sizes()[0],
                "Input temperature and time should have the same number of time steps. The input "
                "time has ",
                _time.sizes()[0],
                " time steps, while the input temperature has ",
                _temperature.sizes()[0],
                " time steps");
    neml_assert(_time.sizes()[1] == _temperature.sizes()[1],
                "Input temperature and time should have the same batch size. The input time has a "
                "batch size of ",
                _time.sizes()[1],
                " while the input temperature has a batch size of ",
                _temperature.sizes()[1]);
  }

  if (_control == "MIXED")
  {
    neml_assert(
        _control_signal.batch_dim() == 2,
        "Input control signal should have 2 batch dimensions but instead has batch dimension",
        _control_signal.batch_dim());
    neml_assert(
        _control_signal.sizes()[0] == _time.sizes()[0],
        "Input control signal should have the same number of steps steps as time, but instead has",
        _control_signal.sizes()[0],
        "time steps");
    neml_assert(
        _control_signal.sizes()[1] == _time.sizes()[1],
        "Input control signal should have the same batch size as time, but instead has batch size",
        _control_signal.sizes()[1]);
  }
}

void
SolidMechanicsDriver::update_forces()
{
  TransientDriver::update_forces();

  auto current_driving_force = _driving_force.batch_index({_step_count});
  _in.base_index_put_(_driving_force_name, current_driving_force);

  if (_temperature_prescribed)
  {
    auto current_temperature = _temperature.batch_index({_step_count});
    _in.base_index_put_(_temperature_name, current_temperature);
  }

  if (_control == "MIXED")
  {
    auto current_control = _control_signal.batch_index({_step_count});
    _in.base_index_put_(_control_name, current_control);
  }
}
}
