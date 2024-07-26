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
#include "neml2/base/guards.h"

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
  for (auto var : _model.output_axis().subaxis("residual").variable_names())
    declare_output_variable(_model.output_axis().subaxis("residual").storage_size(var),
                            _model.output_variable(var.prepend("residual"))->type(),
                            var.prepend("state"));
}

void
ImplicitUpdate::check_AD_limitation() const
{
  neml_assert_dbg(!using_AD_1st_derivative() && !using_AD_2nd_derivative(),
                  "ImplicitUpdate does not support AD because it uses in-place operations to "
                  "iteratively update the trial solution until convergence.");
}

void
ImplicitUpdate::setup_output_views()
{
  Model::setup_output_views();

  if (requires_grad())
  {
    _ds_dsn = derivative_storage().base_index({"state", "old_state"});
    _ds_df = derivative_storage().base_index({"state", "forces"});
    _ds_dfn = derivative_storage().base_index({"state", "old_forces"});
  }
}

void
ImplicitUpdate::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "This model does not define the second derivatives.");

  // Apply initial guess
  LabeledVector sol0(_model.solution(), {&output_axis()});
  sol0.fill(host<VariableStore>()->input_storage());
  if (sol0.tensor().requires_grad())
    sol0.detach_();

  // The trial state is used as the initial guess
  // Perform automatic scaling
  _model.init_scaling(_solver.verbose);

  // Solution
  Tensor sol;

  // Solve for the next state
  {
    SolvingNonlinearSystem solving;
    sol = _model.solution().clone();
    auto [succeeded, iters] = _solver.solve(_model, sol);
    neml_assert(succeeded, "Nonlinear solve failed.");
  }

  if (out)
    output_storage().copy_(sol);

  // Use the implicit function theorem (IFT) to calculate the other derivatives
  if (dout_din)
  {
    // IFT requires dresidual/dinput evaluated at the solution:
    _model.prepare();
    _model.dvalue();
    auto && [dr_ds, dr_dsn, dr_df, dr_dfn] = _model.get_system_matrices();

    // The actual IFT:
    auto [LU, pivot] = math::linalg::lu_factor(dr_ds);
    _ds_dsn.index_put_({torch::indexing::Slice()}, -math::linalg::lu_solve(LU, pivot, dr_dsn));
    _ds_df.index_put_({torch::indexing::Slice()}, -math::linalg::lu_solve(LU, pivot, dr_df));
    _ds_dfn.index_put_({torch::indexing::Slice()}, -math::linalg::lu_solve(LU, pivot, dr_dfn));
  }
}
} // namespace neml2
