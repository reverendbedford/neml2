#pragma once

#include "HardeningMap.h"
#include "InelasticFlowRate.h"
#include "InelasticHardening.h"
#include "YieldSurface.h"

class AssociativeInelasticHardening : public InelasticHardening
{
public:
  AssociativeInelasticHardening(const YieldSurface & surface,
                                HardeningMap & map,
                                InelasticFlowRate & rate);

  /// Output (state_rate)
  virtual StateInfo output() const;

  /// State variables
  virtual StateInfo state() const;
  /// Setup state
  virtual void initial_state(State & input) const;

  /// The rate of the internal variables
  virtual State value(State input);
  /// The derivative of the rate of the internal variables wrt state
  virtual StateDerivative dvalue(State input);

protected:
  const YieldSurface & _surface;
  HardeningMap & _map;
  InelasticFlowRate & _rate;
};
