#pragma once

#include "models/solid_mechanics/HardeningMap.h"
#include "models/solid_mechanics/InelasticFlowRate.h"
#include "tensors/Scalar.h"
#include "models/solid_mechanics/YieldSurface.h"

class PerzynaInelasticFlowRate : public InelasticFlowRate
{
public:
  /// Construct given the value of eta and n
  PerzynaInelasticFlowRate(Scalar eta, Scalar n, const YieldSurface & surface, HardeningMap & map);

  /// The value of the flow rate
  virtual State value(State input);
  /// The derivative of the value of the flow rate with respect to state
  virtual StateDerivative dvalue(State input);

protected:
  Scalar _eta, _n;
  const YieldSurface & _surface;
  HardeningMap & _map;
};
