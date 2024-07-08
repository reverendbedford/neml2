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
  options.doc() =
      "Update an implicit model by solving the underlying implicit system of equations.";

  options.set<std::string>("implicit_model");
  options.set("implicit_model").doc() =
      "The implicit model defining the implicit system of equations to be solved";

  options.set<std::string>("solver");
  options.set("solver").doc() = "Solver used to solve the implicit system";

  return options;
}

ImplicitUpdate::ImplicitUpdate(const OptionSet & options)
  : Model(options),
    _model(register_model<Model>(options.get<std::string>("implicit_model"),
                                 /*extra_deriv_order=*/requires_grad() ? 0 : 1,
                                 /*nonlinear=*/true)),
    _solver(Factory::get_object<NonlinearSolver>("Solvers", options.get<std::string>("solver")))
{
  // Make sure the nonlinear system is square
  neml_assert(_model.input_axis().has_subaxis("state"),
              "The implicit model's input should have a state subaxis. The input axis is\n",
              _model.input_axis());
  neml_assert(_model.output_axis().has_subaxis("residual"),
              "The implicit model's output should have a residual subaxis. The output axis is\n",
              _model.output_axis());
  neml_assert(_model.input_axis().subaxis("state") == _model.output_axis().subaxis("residual"),
              "The implicit model should have conformal trial state and residual. The input state "
              "subaxis is\n",
              _model.input_axis().subaxis("state"),
              "\nThe output residual subaxis is\n",
              _model.output_axis().subaxis("residual"));

  // Take care of dependency registration:
  //   1. Input variables of the "implicit_model" should be *consumed* by *this* model. This has
  //      already been taken care of by the `register_model` call.
  //   2. Output variables of the "implicit_model" on the "residual" subaxis should be *provided* by
  //      *this* model.
  for (auto var : _model.output_axis().subaxis("residual").variable_accessors(/*recursive=*/true))
    declare_output_variable(_model.output_axis().subaxis("residual").storage_size(var),
                            var.on("state"));
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

  // Apply initial condition
  _model.solution() = host<VariableStore>()->input_storage().get({"state"});

  // The trial state is used as the initial guess
  // Perform automatic scaling
  _model.init_scaling(_solver.verbose);

  if (out || dout_din)
  {
    // Solve for the next state
    Model::stage = Model::Stage::SOLVING;
    auto sol = _model.solution().clone();
    auto [succeeded, iters] = _solver.solve(_model, sol);
    neml_assert(succeeded, "Nonlinear solve failed.");
    Model::stage = Model::Stage::UPDATING;

    if (out)
      output_storage().copy_(sol);

    // Use the implicit function theorem (IFT) to calculate the other derivatives
    if (dout_din)
    {
      // IFT requires dresidual/dinput evaluated at the solution:
      _model.value_and_dvalue();
      const auto & partials = _model.derivative_storage();
      // TODO: The following could be views
      const auto dr_ds = partials.get({"residual", "state"});
      const auto dr_dsn = partials.get({"residual", "old_state"});
      const auto dr_df = partials.get({"residual", "forces"});
      const auto dr_dfn = partials.get({"residual", "old_forces"});

      // The actual IFT:
      // TODO: The following could use views
      auto [LU, pivot] = math::linalg::lu_factor(dr_ds);
      derivative_storage().set_({"state", "old_state"}, -math::linalg::lu_solve(LU, pivot, dr_dsn));
      derivative_storage().set_({"state", "forces"}, -math::linalg::lu_solve(LU, pivot, dr_df));
      derivative_storage().set_({"state", "old_forces"},
                                -math::linalg::lu_solve(LU, pivot, dr_dfn));
    }
  }
}
} // namespace neml2
