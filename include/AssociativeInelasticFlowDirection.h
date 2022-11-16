#pragma once

#include "HardeningMap.h"
#include "InelasticFlowDirection.h"
#include "YieldSurface.h"

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
