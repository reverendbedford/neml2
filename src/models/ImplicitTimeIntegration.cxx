#include "models/ImplicitTimeIntegration.h"

ImplicitTimeIntegration::ImplicitTimeIntegration(const std::string & name,
                                                 std::shared_ptr<Model> rate)
  : ImplicitModel(name),
    _rate(*rate)
{
  registerModel(rate);

  // Does the implicit constitutive model already requires time and old time?
  // Probably not, so we add time and old time here anyways, since we need them to perform time
  // integration.
  input().add<LabeledAxis>("forces");
  input().subaxis("forces").add<Scalar>("time");

  input().add<LabeledAxis>("old_forces");
  input().subaxis("old_forces").add<Scalar>("time");

  // Since we are integrating the state in time, we need the old state.
  // The items in old_state should just mirror the state
  input().add<LabeledAxis>("old_state");
  input().subaxis("old_state").merge(input().subaxis("state"));

  // How do we setup the residual so that we can do automatic scaling??
  output().add("residual", _rate.output().storage_size());

  setup();
}

void
ImplicitTimeIntegration::set_value(LabeledVector in,
                                   LabeledVector out,
                                   LabeledMatrix * dout_din) const
{
  TorchSize nbatch = in.batch_size();

  auto t_np1 = in.slice(0, "forces").get<Scalar>("time");
  auto t_n = in.slice(0, "old_forces").get<Scalar>("time");
  auto dt = t_np1 - t_n;

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
  out.set(r, "residual");

  // Finally finally, compute the Jacobian since we have all the information needed anyways
  if (dout_din)
  {
    auto n_state = output().storage_size("residual");
    auto I = BatchTensor<1>::identity(n_state).expand_batch(nbatch);
    auto ds_dot_ds_np1 = drate_din("state", "state");
    auto dr_ds_np1 = I - ds_dot_ds_np1 * dt;
    dout_din->set(dr_ds_np1, "residual", "state");

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

    dout_din->set(dr_ds_n, "residual", "old_state");
    dout_din->set(dr_df_np1, "residual", "forces");
    dout_din->set(dr_df_n, "residual", "old_forces");

    // The other part of the Jacobian goes into the time column
    // as the residual includes the annoying dt
    dout_din->slice(1, "forces")("residual", "time") -= s_dot.unsqueeze(-1);
    dout_din->slice(1, "old_forces")("residual", "time") += s_dot.unsqueeze(-1);
  }
}

void
ImplicitTimeIntegration::set_residual(BatchTensor<1> x, BatchTensor<1> r, BatchTensor<1> * J) const
{
  TorchSize nbatch = x.batch_sizes()[0];
  LabeledVector in(nbatch, input());
  LabeledVector out(nbatch, output());

  // Fill in the current trial state and cached (fixed) forces, old forces, old state
  in.assemble(_cached_in);
  in.set(x, "state");

  if (J)
  {
    LabeledMatrix dout_din(out, in);
    set_value(in, out, &dout_din);
    J->copy_(dout_din("residual", "state"));
  }
  else
    set_value(in, out);

  r.copy_(out("residual"));
}
