#pragma once

#include "models/Model.h"

/// The plastic strain rate given flow rate and flow direction
class PlasticStrainRate : public Model
{
public:
  PlasticStrainRate(const std::string & name);

  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
