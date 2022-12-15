#pragma once

#include "models/Model.h"

namespace neml2
{
/// Rate of an external force
template <typename T>
class ForceRate : public Model
{
public:
  ForceRate(const std::string & name);

  const LabeledAxisAccessor force;
  const LabeledAxisAccessor force_n;
  const LabeledAxisAccessor time;
  const LabeledAxisAccessor time_n;
  const LabeledAxisAccessor force_rate;

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
} // namespace neml2
