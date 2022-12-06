#pragma once

#include "models/solid_mechanics/PlasticFlowRate.h"

class PerzynaPlasticFlowRate : public PlasticFlowRate
{
public:
  /// Construct given the value of eta and n
  PerzynaPlasticFlowRate(const std::string & name, Scalar eta, Scalar n);

  /// The flow rate
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

protected:
  Scalar _eta, _n;
};
