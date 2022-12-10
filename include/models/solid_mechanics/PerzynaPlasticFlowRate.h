#pragma once

#include "models/solid_mechanics/PlasticFlowRate.h"

namespace neml2
{
class PerzynaPlasticFlowRate : public PlasticFlowRate
{
public:
  /// Construct given the value of eta and n
  PerzynaPlasticFlowRate(const std::string & name, Scalar eta, Scalar n);

protected:
  /// The flow rate
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  Scalar _eta, _n;
};
} // namespace neml2
