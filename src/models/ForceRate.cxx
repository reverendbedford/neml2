#include "models/ForceRate.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
template <typename T>
ForceRate<T>::ForceRate(const std::string & name)
  : Model(name),
    force(declareInputVariable<T>({"forces", name})),
    force_n(declareInputVariable<T>({"old_forces", name})),
    time(declareInputVariable<Scalar>({"forces", "time"})),
    time_n(declareInputVariable<Scalar>({"old_forces", "time"})),
    force_rate(declareOutputVariable<T>({"forces", name + "_rate"}))
{
  this->setup();
}

template <typename T>
void
ForceRate<T>::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  auto f_np1 = in.get<T>(force);
  auto f_n = in.get<T>(force_n);
  auto t_np1 = in.get<Scalar>(time);
  auto t_n = in.get<Scalar>(time_n);

  auto df = f_np1 - f_n;
  auto dt = t_np1 - t_n;
  auto f_dot = df / dt;

  out.set(f_dot, force_rate);

  if (dout_din)
  {
    auto df_dot_df_np1 = T::identity_map() / dt;
    auto df_dot_df_n = -T::identity_map() / dt;
    auto df_dot_dt_np1 = -df / dt / dt;
    auto df_dot_dt_n = df / dt / dt;

    dout_din->set(df_dot_df_np1, force_rate, force);
    dout_din->set(df_dot_df_n, force_rate, force_n);
    dout_din->set(df_dot_dt_np1, force_rate, time);
    dout_din->set(df_dot_dt_n, force_rate, time_n);
  }
}

template class ForceRate<Scalar>;
template class ForceRate<SymR2>;
} // namespace neml2
