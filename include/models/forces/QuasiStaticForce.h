#pragma once

#include "models/forces/Force.h"

/// Quasistatic external force
template <typename T, bool stateful>
class QuasiStaticForce : public Force<T, stateful>
{
public:
  QuasiStaticForce(const std::string & name);

  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
