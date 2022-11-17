#include "models/solid_mechanics/AssociativeInelasticHardening.h"

AssociativeInelasticHardening::AssociativeInelasticHardening(const YieldSurface & surface,
                                                             HardeningMap & map,
                                                             InelasticFlowRate & rate)
  : _surface(surface),
    _map(map),
    _rate(rate)
{
}

StateInfo
AssociativeInelasticHardening::output() const
{
  return _map.state().add_suffix("_rate");
}

StateInfo
AssociativeInelasticHardening::state() const
{
  return _map.state();
}

void
AssociativeInelasticHardening::initial_state(State & input) const
{
  _map.initial_state(input);
}

State
AssociativeInelasticHardening::value(State input)
{
  Scalar prefactor = -_rate.value(input).get<Scalar>("flow_rate");
  // This is really annoying.  The values are correct but we have the
  // weird associative hardening thing where we remap df/dq to the
  // strain-like internal variables.
  return _surface.df_ds(_map.value(input))
      .get_substate("hardening_interface")
      .scalar_product(prefactor)
      .replace_info(output());
}

StateDerivative
AssociativeInelasticHardening::dvalue(State input)
{
  Scalar prefactor = -_rate.value(input).get<Scalar>("flow_rate");
  StateDerivative dprefactor = -_rate.dvalue(input);

  // Again we need to rename the state names to deal with associative
  // flow
  StateDerivative A = _surface.d2f_ds2(_map.value(input))
                          .slice_left("hardening_interface")
                          .replace_info_left(output())
                          .chain(_map.dvalue(input))
                          .scalar_product(prefactor);

  // This has the same rename issue as above
  StateDerivative B = _surface.df_ds(_map.value(input))
                          .get_substate("hardening_interface")
                          .replace_info(output())
                          .promote_right("flow_rate")
                          .chain(dprefactor);

  return A + B;
}
