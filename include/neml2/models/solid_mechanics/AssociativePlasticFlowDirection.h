#pragma once

#include "neml2/models/solid_mechanics/PlasticFlowDirection.h"
#include "neml2/models/solid_mechanics/YieldFunction.h"

namespace neml2
{
/// Flow direction associated with a yield surface
class AssociativePlasticFlowDirection : public PlasticFlowDirection
{
public:
  AssociativePlasticFlowDirection(const std::string & name,
                                  const std::shared_ptr<YieldFunction> & f);

  const YieldFunction & yield_function;

protected:
  /// The flow direction
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
} // namespace neml2
