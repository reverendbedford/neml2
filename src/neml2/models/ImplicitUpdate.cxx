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
ImplicitUpdate::ImplicitUpdate(const std::string & name,
                               std::shared_ptr<ImplicitModel> model,
                               std::shared_ptr<NonlinearSolver> solver)
  : Model(name),
    _model(*model),
    _solver(*solver)
{
  register_model(model);
  // Now that the implicit model has been registered, the input of this ImplicitUpdate model should
  // be the same as the implicit model's input. The input subaxes of the implicit model looks
  // something like
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // inputs: state (this is actually the trial state)
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
  output().subaxis("state").merge(model->input().subaxis("state"));
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

  setup();
}

void
ImplicitUpdate::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  TorchSize nbatch = in.batch_size();

  // Cache the input as we are solving the implicit model with FIXED forces, old forces, and old
  // state
  _model.cache_input(in);

  // Solve for the next state
  ImplicitModel::stage = ImplicitModel::Stage::SOLVING;
  BatchTensor<1> sol = _solver.solve(_model, _model.initial_guess(in));
  ImplicitModel::stage = ImplicitModel::Stage::UPDATING;

  out.set(sol, "state");

  // Use the implicit function theorem to calculate the other derivatives
  if (dout_din)
  {
    LabeledVector implicit_in(nbatch, _model.input());
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
} // namespace neml2
