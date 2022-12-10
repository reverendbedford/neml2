#pragma once

#include "models/Model.h"

namespace neml2
{
template <bool rate>
class ElasticStrainTempl : public Model
{
public:
  ElasticStrainTempl(const std::string & name);

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};

typedef ElasticStrainTempl<true> ElasticStrainRate;
typedef ElasticStrainTempl<false> ElasticStrain;
} // namespace neml2
