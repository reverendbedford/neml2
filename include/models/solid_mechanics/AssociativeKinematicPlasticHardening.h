#pragma once

#include "models/solid_mechanics/PlasticHardening.h"
#include "models/solid_mechanics/YieldFunction.h"

namespace neml2
{
/// Equivalent plastic rate associated with a yield surface
class AssociativeKinematicPlasticHardening : public PlasticHardening
{
public:
  AssociativeKinematicPlasticHardening(const std::string & name,
                                       const std::shared_ptr<YieldFunction> & f);

  const YieldFunction & yield_function;

protected:
  /// The flow direction
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

protected:
  /// Accessor for the plastic strain rate
  const LabeledAxisAccessor plastic_strain_rate;
};
} // namespace neml2
