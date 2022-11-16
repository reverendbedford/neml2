#include "AssociativeInelasticFlowDirection.h"

AssociativeInelasticFlowDirection::AssociativeInelasticFlowDirection(const YieldSurface & surface,
                                                                     HardeningMap & map)
  : _surface(surface),
    _map(map)
{
}

State
AssociativeInelasticFlowDirection::value(State input)
{
  State res = State::same_batch(output(), input);
  res.set<SymR2>("flow_direction", _surface.df_ds(_map.value(input)).get<SymR2>("stress"));
  return res;
}

StateDerivative
AssociativeInelasticFlowDirection::dvalue(State input)
{
  return _surface.d2f_ds2(_map.value(input))
      .slice_left("stress_interface")
      .replace_info_left(output())
      .chain(_map.dvalue(input));
}
