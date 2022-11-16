#pragma once

#include "HardeningMap.h"
#include "InelasticFlowRate.h"
#include "Scalar.h"
#include "YieldSurface.h"

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
