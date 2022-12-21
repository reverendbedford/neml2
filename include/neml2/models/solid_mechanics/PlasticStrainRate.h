#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
/// The plastic strain rate given flow rate and flow direction
class PlasticStrainRate : public Model
{
public:
  PlasticStrainRate(const std::string & name);

  const LabeledAxisAccessor hardening_rate;
  const LabeledAxisAccessor plastic_flow_direction;
  const LabeledAxisAccessor plastic_strain_rate;

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
} // namespace neml2
