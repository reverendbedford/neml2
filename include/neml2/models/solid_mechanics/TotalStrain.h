#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
template <bool rate>
class TotalStrainTempl : public Model
{
public:
  TotalStrainTempl(const std::string & name);

  const LabeledAxisAccessor elastic_strain;
  const LabeledAxisAccessor plastic_strain;
  const LabeledAxisAccessor total_strain;

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};

typedef TotalStrainTempl<true> TotalStrainRate;
typedef TotalStrainTempl<false> TotalStrain;
} // namespace neml2
