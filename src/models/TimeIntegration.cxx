#include "models/TimeIntegration.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
template <typename T>
TimeIntegration<T>::TimeIntegration(const std::string & name)
  : Model(name)
{
  this->input().template add<LabeledAxis>("state");
  this->input().subaxis("state").template add<T>(name + "_rate");

  this->input().template add<LabeledAxis>("old_state");
  this->input().subaxis("old_state").template add<T>(name);

  // We need time to perform time integration
  this->input().template add<LabeledAxis>("forces");
  this->input().subaxis("forces").template add<Scalar>("time");

  this->input().template add<LabeledAxis>("old_forces");
  this->input().subaxis("old_forces").template add<Scalar>("time");

  this->output().template add<LabeledAxis>("state");
  this->output().subaxis("state").template add<T>(name);

  this->setup();
}

template <typename T>
void
TimeIntegration<T>::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  auto s_dot = in.slice("state").get<T>(this->name() + "_rate");
  auto s_n = in.slice("old_state").get<T>(this->name());
  auto t_np1 = in.slice("forces").get<Scalar>("time");
  auto t_n = in.slice("old_forces").get<Scalar>("time");
  auto dt = t_np1 - t_n;

  // s_np1 = s_n + s_dot * (t_np1 - t_n)
  auto s_np1 = s_n + s_dot * dt;
  out.set(s_np1, "state");

  // Finally, compute the Jacobian since we have all the information needed anyways
  if (dout_din)
  {
    auto ds_np1_ds_dot = T::identity_map() / dt;
    auto ds_np1_ds_n = T::identity_map();
    auto ds_np1_dt_np1 = s_dot;
    auto ds_np1_dt_n = -s_dot;

    dout_din->block("state", "state").set(ds_np1_ds_dot, this->name(), this->name() + "_rate");
    dout_din->block("state", "old_state").set(ds_np1_ds_n, this->name(), this->name());
    dout_din->block("state", "forces").set(ds_np1_dt_np1, this->name(), "time");
    dout_din->block("state", "old_forces").set(ds_np1_dt_n, this->name(), "time");
  }
}

template class TimeIntegration<Scalar>;
template class TimeIntegration<SymR2>;
} // namespace neml2
