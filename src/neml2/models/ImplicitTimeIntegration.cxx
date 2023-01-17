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

#include "neml2/models/ImplicitTimeIntegration.h"

namespace neml2
{
ImplicitTimeIntegration::ImplicitTimeIntegration(const std::string & name,
                                                 std::shared_ptr<Model> rate)
  : ImplicitModel(name),
    time(declareInputVariable<Scalar>({"forces", "time"})),
    time_n(declareInputVariable<Scalar>({"old_forces", "time"})),
    resid(declareOutputVariable(rate->output().storage_size(), {"residual"})),
    _rate(*rate)
{
  register_model(rate);

  // Since we are integrating the state in time, we need the old state.
  // The items in old_state should just mirror the state
  // TODO: simplify the following code
  input().add<LabeledAxis>("old_state");
  auto merged_vars = input().subaxis("old_state").merge(input().subaxis("state"));
  for (auto & merged_var : merged_vars)
  {
    merged_var.item_names.insert(merged_var.item_names.begin(), "old_state");
    _consumed_vars.insert(merged_var);
  }

  setup();
}

void
ImplicitTimeIntegration::set_value(LabeledVector in,
                                   LabeledVector out,
                                   LabeledMatrix * dout_din) const
{
  TorchSize nbatch = in.batch_size();
  auto dt = in.get<Scalar>(time) - in.get<Scalar>(time_n);

  // First evaluate the rate model AND optionally its derivatives
  LabeledVector rate(nbatch, _rate.output());
  LabeledMatrix drate_din(nbatch, _rate.output(), input());
  if (dout_din)
    std::tie(rate, drate_din) = _rate.value_and_dvalue(in);
  else
    rate = _rate.value(in);

  // r = s_np1 - s_n - s_dot * (t_np1 - t_n)
  auto s_np1 = in("state");
  auto s_n = in("old_state");
  auto s_dot = rate("state");

  // Finally, the residual
  auto r = s_np1 - s_n - s_dot * dt;
  out.set(r, resid);

  // Finally finally, compute the Jacobian since we have all the information needed anyways
  if (dout_din)
  {
    auto n_state = resid.storage_size;
    auto I = BatchTensor<1>::identity(n_state).batch_expand(nbatch);
    auto ds_dot_ds_np1 = drate_din("state", "state");
    auto dr_ds_np1 = I - ds_dot_ds_np1 * dt;
    dout_din->set(dr_ds_np1, resid, "state");

    // While solving the implicit model we only care about dresidual/dstate
    if (ImplicitModel::stage == ImplicitModel::Stage::SOLVING)
      return;

    auto ds_dot_ds_n = drate_din("state", "old_state");
    auto ds_dot_df_np1 = drate_din("state", "forces");
    auto ds_dot_df_n = drate_din("state", "old_forces");

    // (Part of) the actual Jacobian
    auto dr_ds_n = -I - ds_dot_ds_n * dt;
    auto dr_df_np1 = -ds_dot_df_np1 * dt;
    auto dr_df_n = -ds_dot_df_n * dt;

    dout_din->set(dr_ds_n, resid, "old_state");
    dout_din->set(dr_df_np1, resid, "forces");
    dout_din->set(dr_df_n, resid, "old_forces");

    // The other part of the Jacobian goes into the time column
    // as the residual includes the annoying dt
    (*dout_din)(resid, time) -= s_dot.unsqueeze(-1);
    (*dout_din)(resid, time_n) += s_dot.unsqueeze(-1);
  }
}

void
ImplicitTimeIntegration::set_residual(BatchTensor<1> x, BatchTensor<1> r, BatchTensor<1> * J) const
{
  TorchSize nbatch = x.batch_sizes()[0];
  LabeledVector in(nbatch, input());
  LabeledVector out(nbatch, output());

  // Fill in the current trial state and cached (fixed) forces, old forces, old state
  in.fill(_cached_in);
  in.set(x, "state");

  if (J)
  {
    LabeledMatrix dout_din(out, in);
    set_value(in, out, &dout_din);
    J->copy_(dout_din(resid, "state"));
  }
  else
    set_value(in, out);

  r.copy_(out(resid));
}
} // namespace neml2
