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

#include "neml2/models/solid_mechanics/YieldFunction.h"
#include "neml2/tensors/LabeledAxis.h"
#include "neml2/tensors/LabeledVector.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
YieldFunction::YieldFunction(const std::string & name,
                             const std::shared_ptr<StressMeasure> & sm,
                             Scalar s0,
                             bool with_isotropic_hardening,
                             bool with_kinematic_hardening)
  : SecDerivModel(name),
    mandel_stress(declareInputVariable<SymR2>({"state", "mandel_stress"})),
    yield_function(declareOutputVariable<Scalar>({"state", "yield_function"})),
    isotropic_hardening(
        with_isotropic_hardening
            ? declareInputVariable<Scalar>({"state", "hardening_interface", "isotropic_hardening"})
            : LabeledAxisAccessor({})),
    kinematic_hardening(
        with_kinematic_hardening
            ? declareInputVariable<SymR2>({"state", "hardening_interface", "kinematic_hardening"})
            : LabeledAxisAccessor({})),
    stress_measure(*sm),
    _s0(register_parameter("yield_stress", s0)),
    _with_isotropic_hardening(with_isotropic_hardening),
    _with_kinematic_hardening(with_kinematic_hardening)
{
  register_model(sm, false);
  setup();
}

void
YieldFunction::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  // Make input for the stress measure
  auto sm_input = make_stress_measure_input(in);

  // Actually calculate the stress measure
  auto m = stress_measure.value(sm_input).slice("state").get<Scalar>("stress_measure");

  // Calculate the perfectly plastic part
  auto f = m - sqrt(2.0 / 3.0) * _s0;

  // Calculate the isotropic hardening part
  if (_with_isotropic_hardening)
    f -= sqrt(2.0 / 3.0) * in.get<Scalar>(isotropic_hardening);

  // Set the output
  out.set(f, yield_function);

  if (dout_din)
  {
    auto dm = stress_measure.dvalue(sm_input)
                  .slice(0, "state")
                  .slice(1, "state")
                  .get<SymR2>("stress_measure", "overstress");
    // Derivative wrt. mandel stress
    dout_din->set(dm, yield_function, mandel_stress);

    // Derivative wrt. isotropic hardening
    if (_with_isotropic_hardening)
    {
      auto df_dg = Scalar(-sqrt(2.0 / 3.0), in.batch_size());
      dout_din->set(df_dg, yield_function, isotropic_hardening);
    }

    // Derivative wrt. kinematic hardening
    if (_with_kinematic_hardening)
    {
      dout_din->set(-dm, yield_function, kinematic_hardening);
    }
  }
}

void
YieldFunction::set_dvalue(LabeledVector in,
                          LabeledMatrix dout_din,
                          LabeledTensor<1, 3> * d2out_din2) const
{
  // Make input for the stress measure
  auto sm_input = make_stress_measure_input(in);

  // Get the derivative of the stress measure wrt Mandel stress
  SymR2 dm;
  LabeledMatrix DM;
  LabeledTensor<1, 3> DM2;
  if (d2out_din2)
  {
    std::tie(DM, DM2) = stress_measure.dvalue_and_d2value(sm_input);
    dm = DM.slice(0, "state").slice(1, "state").get<SymR2>("stress_measure", "overstress");
  }
  else
  {
    dm = stress_measure.dvalue(sm_input)
             .slice(0, "state")
             .slice(1, "state")
             .get<SymR2>("stress_measure", "overstress");
  }

  // Derivative wrt. mandel stress
  dout_din.set(dm, yield_function, mandel_stress);

  // Derivative wrt. isotropic hardening
  if (_with_isotropic_hardening)
  {
    auto df_dg = Scalar(-sqrt(2.0 / 3.0), in.batch_size());
    dout_din.set(df_dg, yield_function, isotropic_hardening);
  }

  // Derivative wrt. kinematic hardening
  if (_with_kinematic_hardening)
  {
    dout_din.set(-dm, yield_function, kinematic_hardening);
  }

  if (d2out_din2)
  {
    // Perfectly plastic part: yield, mandel, mandel
    SymSymR4 ymm = DM2.slice(0, "state")
                       .slice(1, "state")
                       .slice(2, "state")
                       .get<SymSymR4>("stress_measure", "overstress", "overstress");
    d2out_din2->set(ymm, yield_function, mandel_stress, mandel_stress);

    // All the isotropic hardening 2nd derivatives are zero

    // Kinematic term is pretty easy...
    if (_with_kinematic_hardening)
    {
      d2out_din2->set(-ymm, yield_function, mandel_stress, kinematic_hardening);
      d2out_din2->set(ymm, yield_function, kinematic_hardening, kinematic_hardening);
      d2out_din2->set(-ymm, yield_function, kinematic_hardening, mandel_stress);
    }
  }
}

LabeledVector
YieldFunction::make_stress_measure_input(LabeledVector in) const
{
  // First retrieve the hardening variables
  auto mandel = in.get<SymR2>(mandel_stress);

  // Calculate the overstress
  auto overstress = mandel.clone();
  if (_with_kinematic_hardening)
    overstress -= in.get<SymR2>(kinematic_hardening);

  // Get the input to the stress measure
  TorchSize nbatch = in.batch_size();
  LabeledVector sm_input(nbatch, stress_measure.input());
  sm_input.slice("state").set(overstress, "overstress");

  return sm_input;
}

} // namespace neml2
