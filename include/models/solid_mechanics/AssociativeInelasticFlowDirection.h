#pragma once

#include "models/solid_mechanics/HardeningMap.h"
#include "models/solid_mechanics/InelasticFlowDirection.h"
#include "models/solid_mechanics/YieldSurface.h"

/// Flow direction associated with a yield surface
class AssociativeInelasticFlowDirection : public InelasticFlowDirection
{
public:
  AssociativeInelasticFlowDirection(const YieldSurface & surface, HardeningMap & map);

  /// Define the flow direction
  virtual State value(State input);
  /// Derivative of the flow direction with respect to stress
  virtual StateDerivative dvalue(State input);

protected:
  const YieldSurface & _surface;
  HardeningMap & _map;
};
