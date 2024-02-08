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

#include "neml2/models/ImplicitUpdate.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(ImplicitUpdate);

OptionSet
ImplicitUpdate::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<std::string>("implicit_model");
  options.set<std::string>("solver");
  return options;
}

ImplicitUpdate::ImplicitUpdate(const OptionSet & options)
  : Model(options),
    _model(register_model<Model>(options.get<std::string>("implicit_model"))),
    _solver(Factory::get_object<NonlinearSolver>("Solvers", options.get<std::string>("solver")))
{
  // Take care of dependency registration:
  //   1. Input variables of the "implicit_model" should be *consumed* by *this* model. This has
  //      already been taken care of by the `register_model` call.
  //   2. Output variables of the "implicit_model" on the "residual" subaxis should be *provided* by
  //      *this* model.
  for (auto var : _model.output_axis().subaxis("residual").variable_accessors(/*recursive=*/true))
    declare_output_variable(var.on("state"),
                            _model.output_axis().subaxis("residual").storage_size(var));
}

void
ImplicitUpdate::check_AD_limitation() const
{
  neml_assert_dbg(!_AD_1st_deriv && !_AD_2nd_deriv,
                  "ImplicitUpdate does not support AD because it uses in-place operations to "
                  "iteratively update the trial solution until convergence.");
}

void
ImplicitUpdate::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "This model does not define the second derivatives.");
  neml_assert_dbg(
      !dout_din || out,
      "ImplicitUpdate: requires the value and the first derivatives to be computed together.");

  // The trial state is used as the initial guess
  // Perform automatic scaling
  _model.init_scaling(_solver.verbose);

  if (out || dout_din)
  {
    // Solve for the next state
    Model::stage = Model::Stage::SOLVING;
    auto [succeeded, iters] = _solver.solve(_model);
    neml_assert(succeeded, "Nonlinear solve failed.");
    output_storage().copy_(_model.solution());
    Model::stage = Model::Stage::UPDATING;

    // Use the implicit function theorem (IFT) to calculate the other derivatives
    if (dout_din)
    {
      // IFT requires dstate/dinput evaluated at the solution:
      _model.value_and_dvalue();
      auto partials = _model.get_doutput_dinput();

      // The actual IFT:
      LabeledMatrix J = partials.slice(1, "state");
      auto [LU, pivot] = math::linalg::lu_factor(J);
      derivative_storage()("state", "old_state")
          .copy_(-math::linalg::lu_solve(LU, pivot, partials.slice(1, "old_state")));
      derivative_storage()("state", "forces")
          .copy_(-math::linalg::lu_solve(LU, pivot, partials.slice(1, "forces")));
      derivative_storage()("state", "old_forces")
          .copy_(-math::linalg::lu_solve(LU, pivot, partials.slice(1, "old_forces")));
    }
  }
}
} // namespace neml2
