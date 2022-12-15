#include "models/TimeIntegration.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
template <typename T>
TimeIntegration<T>::TimeIntegration(const std::string & name)
  : Model(name),
    var_rate(declareInputVariable<T>("state", name + "_rate")),
    var_n(declareInputVariable<T>("old_state", name)),
    time(declareInputVariable<Scalar>("forces", "time")),
    time_n(declareInputVariable<Scalar>("old_forces", "time")),
    var(declareOutputVariable<T>("state", name))
{
  this->setup();
}

template <typename T>
void
TimeIntegration<T>::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  auto s_dot = in.get<T>(var_rate);
  auto s_n = in.get<T>(var_n);
  auto t_np1 = in.get<Scalar>(time);
  auto t_n = in.get<Scalar>(time_n);
  auto dt = t_np1 - t_n;

  // s_np1 = s_n + s_dot * (t_np1 - t_n)
  auto s_np1 = s_n + s_dot * dt;
  out.set(s_np1, var);

  // Finally, compute the Jacobian since we have all the information needed anyways
  if (dout_din)
  {
    auto ds_np1_ds_dot = T::identity_map() / dt;
    auto ds_np1_ds_n = T::identity_map();
    auto ds_np1_dt_np1 = s_dot;
    auto ds_np1_dt_n = -s_dot;

    dout_din->set(ds_np1_ds_dot, var, var_rate);
    dout_din->set(ds_np1_ds_n, var, var_n);
    dout_din->set(ds_np1_dt_np1, var, time);
    dout_din->set(ds_np1_dt_n, var, time_n);
  }
}

template class TimeIntegration<Scalar>;
template class TimeIntegration<SymR2>;
} // namespace neml2
