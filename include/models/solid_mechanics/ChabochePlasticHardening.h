#pragma once

#include "models/solid_mechanics/PlasticHardening.h"
#include "models/solid_mechanics/YieldFunction.h"
#include "tensors/LabeledAxis.h"

namespace neml2
{
class ChabochePlasticHardening : public PlasticHardening
{
public:
  ChabochePlasticHardening(const std::string & name,
                           Scalar C,
                           Scalar g,
                           Scalar A,
                           Scalar a,
                           const std::string backstress_suffix = "");

  /// Input: the current value of this backstress
  const LabeledAxisAccessor backstress;
  /// Input: the current flow direction
  const LabeledAxisAccessor flow_direction;
  /// Output: the rate of this backstress
  const LabeledAxisAccessor backstress_rate;

protected:
  /// Set the rate of the backstress and derivatives, if requested
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  Scalar _C, _g, _A, _a;
};

} // namespace neml2
