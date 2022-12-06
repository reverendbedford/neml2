#pragma once

#include "models/solid_mechanics/PlasticHardening.h"
#include "models/solid_mechanics/YieldFunction.h"

/// Equivalent plastic rate associated with a yield surface
class AssociativePlasticHardening : public PlasticHardening
{
public:
  AssociativePlasticHardening(const std::string & name, YieldFunction & f);

  /// The flow direction
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  YieldFunction & yield_function;
};
