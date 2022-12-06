#pragma once

#include "models/forces/Force.h"

/// Rate of an external force
template <typename T>
class ForceRate : public Force<T, false>
{
public:
  ForceRate(const std::string & name);

  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
