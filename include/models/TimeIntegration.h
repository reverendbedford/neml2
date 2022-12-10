#pragma once

#include "models/Model.h"

namespace neml2
{
/// Perform first order time integration
template <typename T>
class TimeIntegration : public Model
{
public:
  TimeIntegration(const std::string & name);

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
} // namespace neml2
