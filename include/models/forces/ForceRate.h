#pragma once

#include "models/forces/Force.h"

namespace neml2
{
/// Rate of an external force
template <typename T>
class ForceRate : public Force<T, false>
{
public:
  ForceRate(const std::string & name);

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
} // namespace neml2
