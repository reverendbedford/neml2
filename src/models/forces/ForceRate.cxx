#include "models/forces/ForceRate.h"

template <typename T>
ForceRate<T>::ForceRate(const std::string & name)
  : Force<T, false>(name)
{
  this->input().subaxis("forces").template add<T>(name);
  this->input().subaxis("forces").template add<Scalar>("time");

  this->input().template add<LabeledAxis>("old_forces");
  this->input().subaxis("old_forces").template add<T>(name);
  this->input().subaxis("old_forces").template add<Scalar>("time");

  this->output().subaxis("forces").template add<T>(name + "_rate");

  this->setup();
}

template <typename T>
void
ForceRate<T>::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  auto f_np1 = in.slice(0, "forces").get<T>(this->name());
  auto t_np1 = in.slice(0, "forces").get<Scalar>("time");
  auto f_n = in.slice(0, "old_forces").get<T>(this->name());
  auto t_n = in.slice(0, "old_forces").get<Scalar>("time");

  auto df = f_np1 - f_n;
  auto dt = t_np1 - t_n;
  auto f_dot = df / dt;

  out.slice(0, "forces").set(f_dot, this->name() + "_rate");

  if (dout_din)
  {
    auto df_dot_df_np1 = T::identity_map() / dt;
    auto df_dot_df_n = -T::identity_map() / dt;
    auto df_dot_dt_np1 = -df / dt / dt;
    auto df_dot_dt_n = df / dt / dt;

    dout_din->block("forces", "forces").set(df_dot_df_np1, this->name() + "_rate", this->name());
    dout_din->block("forces", "forces").set(df_dot_dt_np1, this->name() + "_rate", "time");
    dout_din->block("forces", "old_forces").set(df_dot_df_n, this->name() + "_rate", this->name());
    dout_din->block("forces", "old_forces").set(df_dot_dt_n, this->name() + "_rate", "time");
  }
}

template class ForceRate<Scalar>;
template class ForceRate<SymR2>;
