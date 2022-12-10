#include "SampleRateModel.h"
#include "tensors/SymSymR4.h"

using namespace neml2;

template <bool is_ad>
SampleRateModelTempl<is_ad>::SampleRateModelTempl(const std::string & name)
  : SampleRateModelBase<is_ad>(name)
{
  this->input().template add<LabeledAxis>("state");
  this->input().subaxis("state").template add<Scalar>("foo");
  this->input().subaxis("state").template add<Scalar>("bar");
  this->input().subaxis("state").template add<SymR2>("baz");

  this->input().template add<LabeledAxis>("forces");
  this->input().subaxis("forces").template add<Scalar>("temperature");

  this->output().template add<LabeledAxis>("state");
  this->output().subaxis("state").template add<Scalar>("foo_rate");
  this->output().subaxis("state").template add<Scalar>("bar_rate");
  this->output().subaxis("state").template add<SymR2>("baz_rate");

  this->setup();
}

template <bool is_ad>
void
SampleRateModelTempl<is_ad>::set_value(LabeledVector in,
                                       LabeledVector out,
                                       LabeledMatrix * dout_din) const
{
  // Grab the trial states
  auto foo = in.slice(0, "state").get<Scalar>("foo");
  auto bar = in.slice(0, "state").get<Scalar>("bar");
  auto baz = in.slice(0, "state").get<SymR2>("baz");

  // Say the rates depend on temperature, for fun
  auto T = in.slice(0, "forces").get<Scalar>("temperature");

  // Some made up rates
  auto foo_dot = (foo * foo + bar) * T + baz.tr();
  auto bar_dot = -bar / 100 - 0.5 * foo - 0.9 * T + baz.tr();
  auto baz_dot = (foo + bar) * baz * (T - 3);

  // Set the output
  out.slice(0, "state").set(foo_dot, "foo_rate");
  out.slice(0, "state").set(bar_dot, "bar_rate");
  out.slice(0, "state").set(baz_dot, "baz_rate");

  if constexpr (!is_ad)
    if (dout_din)
    {
      TorchSize nbatch = in.batch_size();
      auto dfoo_dot_dfoo = 2 * foo * T;
      auto dfoo_dot_dbar = T;
      auto dfoo_dot_dbaz = SymR2::identity().batch_expand(nbatch);
      auto dbar_dot_dfoo = Scalar(-0.5, nbatch);
      auto dbar_dot_dbar = Scalar(-0.01, nbatch);
      auto dbar_dot_dbaz = SymR2::identity().batch_expand(nbatch);
      auto dbaz_dot_dfoo = baz * (T - 3);
      auto dbaz_dot_dbar = baz * (T - 3);
      auto dbaz_dot_dbaz = (foo + bar) * (T - 3) * SymR2::identity_map().batch_expand(nbatch);

      dout_din->block("state", "state").set(dfoo_dot_dfoo, "foo_rate", "foo");
      dout_din->block("state", "state").set(dfoo_dot_dbar, "foo_rate", "bar");
      dout_din->block("state", "state").set(dfoo_dot_dbaz, "foo_rate", "baz");
      dout_din->block("state", "state").set(dbar_dot_dfoo, "bar_rate", "foo");
      dout_din->block("state", "state").set(dbar_dot_dbar, "bar_rate", "bar");
      dout_din->block("state", "state").set(dfoo_dot_dbaz, "bar_rate", "baz");
      dout_din->block("state", "state").set(dbaz_dot_dfoo, "baz_rate", "foo");
      dout_din->block("state", "state").set(dbaz_dot_dbar, "baz_rate", "bar");
      dout_din->block("state", "state").set(dbaz_dot_dbaz, "baz_rate", "baz");

      auto dfoo_dot_dT = foo * foo + bar;
      auto dbar_dot_dT = Scalar(-0.9, nbatch);
      auto dbaz_dot_dT = (foo + bar) * baz;

      dout_din->block("state", "forces").set(dfoo_dot_dT, "foo_rate", "temperature");
      dout_din->block("state", "forces").set(dbar_dot_dT, "bar_rate", "temperature");
      dout_din->block("state", "forces").set(dbaz_dot_dT, "baz_rate", "temperature");
    }
}

template class SampleRateModelTempl<true>;
template class SampleRateModelTempl<false>;
