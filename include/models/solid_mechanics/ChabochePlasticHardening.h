#pragma once

#include "models/solid_mechanics/PlasticHardening.h"
#include "models/solid_mechanics/YieldFunction.h"

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
                           const std::shared_ptr<YieldFunction> & f,
                           const std::string backstress_suffix = "");
  const YieldFunction & yield_function;
  const LabeledAxisAccessor backstress;
  const LabeledAxisAccessor backstress_rate;

protected:
  /// The flow direction
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  Scalar _C, _g, _A, _a;
};

} // namespace neml2
