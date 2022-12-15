#pragma once

#include "models/solid_mechanics/PlasticHardening.h"
#include "models/solid_mechanics/IsotropicYieldFunction.h"

namespace neml2
{
/// Equivalent plastic rate associated with a yield surface
class AssociativePlasticHardening : public PlasticHardening
{
public:
  AssociativePlasticHardening(const std::string & name,
                              const std::shared_ptr<IsotropicYieldFunction> & f);

  const IsotropicYieldFunction & yield_function;

protected:
  /// The flow direction
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
} // namespace neml2
