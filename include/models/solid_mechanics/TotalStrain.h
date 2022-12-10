#pragma once

#include "models/Model.h"

namespace neml2
{
template <bool rate>
class TotalStrainTempl : public Model
{
public:
  TotalStrainTempl(const std::string & name);

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};

typedef TotalStrainTempl<true> TotalStrainRate;
typedef TotalStrainTempl<false> TotalStrain;
} // namespace neml2
