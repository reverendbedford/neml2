#pragma once

#include "neml2/models/solid_mechanics/PlasticHardening.h"
#include "neml2/models/solid_mechanics/YieldFunction.h"

namespace neml2
{
/// Equivalent plastic rate associated with a yield surface
class AssociativeIsotropicPlasticHardening : public PlasticHardening
{
public:
  AssociativeIsotropicPlasticHardening(const std::string & name,
                                       const std::shared_ptr<YieldFunction> & f);

  /// Yield function used to define the hardening rule
  const YieldFunction & yield_function;

  /// Accessor for the plastic strain rate
  const LabeledAxisAccessor equivalent_plastic_strain_rate;

protected:
  /// The flow direction
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
} // namespace neml2
