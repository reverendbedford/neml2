#pragma once

#include "models/solid_mechanics/PlasticFlowDirection.h"
#include "models/solid_mechanics/YieldFunction.h"

/// Flow direction associated with a yield surface
class AssociativePlasticFlowDirection : public PlasticFlowDirection
{
public:
  AssociativePlasticFlowDirection(const std::string & name, YieldFunction & f);

  /// The flow direction
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  YieldFunction & yield_function;
};
