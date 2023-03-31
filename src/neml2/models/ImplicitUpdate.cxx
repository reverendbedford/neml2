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
#include "neml2/predictors/PreviousStatePredictor.h"

namespace neml2
{
register_NEML2_object(ImplicitUpdate);

ParameterSet
ImplicitUpdate::expected_params()
{
  ParameterSet params = Model::expected_params();
  params.set<std::string>("implicit_model");
  params.set<std::string>("solver");
  params.set<std::string>("predictor");
  return params;
}

ImplicitUpdate::ImplicitUpdate(const ParameterSet & params)
  : Model(params),
    _model(Factory::get_object<Model>("Models", params.get<std::string>("implicit_model"))),
    _solver(Factory::get_object<NonlinearSolver>("Solvers", params.get<std::string>("solver")))
{
  register_model(
      Factory::get_object_ptr<Model>("Models", params.get<std::string>("implicit_model")));
  // Now that the implicit model has been registered, the input of this ImplicitUpdate model should
  // be the same as the implicit model's input. The input subaxes of the implicit model looks
  // something like
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // inputs: state (trial state) ---> outputs: residual
  //         old state
  //         forces
  //         old forces
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // However, the inputs and outputs of *this* model (ImplicitUpdate) should look like this after
  // the update:
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // inputs: old state  ------> outputs: state
  //         forces
  //         old forces
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // as we have eliminated the trial state by solving the nonlinear system.
  // So, we need to remove the "state" subaxis from the input and add it to the output
  output().add<LabeledAxis>("state");
  output().subaxis("state").merge(_model.input().subaxis("state"));
  input().remove("state");

  // The consumed vars from "state" become provided vars
  // TODO: generalize and simplify the followng code
  for (auto consumed_var_it = _consumed_vars.begin(); consumed_var_it != _consumed_vars.end();
       consumed_var_it++)
    if (consumed_var_it->item_names[0] == "state")
    {
      _provided_vars.insert(*consumed_var_it);
      _consumed_vars.erase(consumed_var_it--);
    }

  // Add predictor. If no predictor is specified by the user, default to PreviousStatePredictor
  const auto predictor_name = params.get<std::string>("predictor");
  if (predictor_name.empty())
  {
    Factory::get_factory().create_object(
        "Predictors",
        PreviousStatePredictor::expected_params() +
            ParameterSet(KS{"name", "_default_predictor"}, KS{"type", "PreviousStatePredictor"}));
    _predictor = Factory::get_object_ptr<Predictor>("Predictors", "_default_predictor").get();
  }
  else
    _predictor = Factory::get_object_ptr<Predictor>("Predictors", predictor_name).get();

  setup();
}

LabeledMatrix
ImplicitUpdate::dvalue(LabeledVector in) const
{
  auto [out, dout_din] = Model::value_and_dvalue(in);
  return dout_din;
}

void
ImplicitUpdate::set_value(LabeledVector in,
                          LabeledVector * out,
                          LabeledMatrix * dout_din,
                          LabeledTensor3D * d2out_din2) const
{
  neml_assert_dbg(!d2out_din2, "This model does not define the second derivatives.");

  const auto options = in.options();
  const auto nbatch = in.batch_size();

  // Cache the input as we are solving the implicit model with FIXED forces, old forces, and old
  // state
  _model.cache_input(in);

  // Set the initial guess
  _predictor->set_initial_guess(in, *out);

  // Solve for the next state
  Model::stage = Model::Stage::SOLVING;
  BatchTensor<1> sol = _solver.solve(_model, (*out)("state"));
  Model::stage = Model::Stage::UPDATING;

  if (out)
  {
    out->set(sol, "state");
    _predictor->post_solve(in, *out);
  }

  // Use the implicit function theorem to calculate the other derivatives
  if (dout_din)
  {
    LabeledVector implicit_in(nbatch, {&_model.input()}, options);
    implicit_in.fill(in);
    implicit_in.set(sol, "state");

    auto partials = _model.dvalue(implicit_in);
    LabeledMatrix J = partials.slice(1, "state");
    LabeledMatrix Jinv = J.inverse();
    dout_din->block("state", "old_state").copy(-Jinv.chain(partials.slice(1, "old_state")));
    dout_din->block("state", "forces").copy(-Jinv.chain(partials.slice(1, "forces")));
    dout_din->block("state", "old_forces").copy(-Jinv.chain(partials.slice(1, "old_forces")));
  }
}

void
ImplicitUpdate::advance_step()
{
  Model::advance_step();
  _predictor->advance_step();
}
} // namespace neml2
