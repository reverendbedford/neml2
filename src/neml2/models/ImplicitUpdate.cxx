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
    _model(include_model<Model>(options.get<std::string>("implicit_model"))),
    _solver(Factory::get_object<NonlinearSolver>("Solvers", options.get<std::string>("solver")))
{
  // Now that the implicit model has been registered, the input of this ImplicitUpdate model should
  // be the same as the implicit model's input. The input subaxes of the implicit model looks
  // something like
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // inputs: state (trial) -------> outputs: residual
  //         old state
  //         forces
  //         old forces
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // The inputs and outputs of *this* model (ImplicitUpdate) should look like this after
  // the update:
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // inputs: state (trial)  ------> outputs: state
  //         old state
  //         forces
  //         old forces
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // So, we need to add the state to the output
  output().add<LabeledAxis>("state");
  for (auto var : _model.output().subaxis("residual").variable_accessors(/*recursive=*/true))
  {
    auto sz = _model.output().subaxis("residual").storage_size(var);
    declare_output_variable(var.on("state"), sz);
  }

  setup();
}

void
ImplicitUpdate::set_value(const LabeledVector & in,
                          LabeledVector * out,
                          LabeledMatrix * dout_din,
                          LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "This model does not define the second derivatives.");
  neml_assert_dbg(
      !dout_din || out,
      "ImplicitUpdate: requires the value and the first derivatives to be computed together.");

  // Cache the input as we are solving the implicit model with FIXED forces, old forces, and old
  // state
  _model.cache_input(in);

  // The trial state is used as the initial guess
  auto guess = in("state");

  if (out || dout_din)
  {
    // Solve for the next state
    Model::stage = Model::Stage::SOLVING;
    BatchTensor sol = _solver.solve(_model, guess);
    Model::stage = Model::Stage::UPDATING;

    if (out)
      out->set(sol, "state");

    // Use the implicit function theorem to calculate the other derivatives
    if (dout_din)
    {
      auto implicit_in = LabeledVector::empty(in.batch_sizes(), {&_model.input()}, in.options());
      implicit_in.fill(in);
      implicit_in.set(sol, "state");

      auto partials = _model.dvalue(implicit_in);
      LabeledMatrix J = partials.slice(1, "state");
      auto [LU, pivot] = torch::linalg_lu_factor(J);
      dout_din->block("state", "old_state")
          .copy_(-torch::linalg_lu_solve(LU, pivot, partials.slice(1, "old_state")));
      dout_din->block("state", "forces")
          .copy_(-torch::linalg_lu_solve(LU, pivot, partials.slice(1, "forces")));
      dout_din->block("state", "old_forces")
          .copy_(-torch::linalg_lu_solve(LU, pivot, partials.slice(1, "old_forces")));
    }
  }
}
} // namespace neml2
