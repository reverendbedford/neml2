#pragma once

#include "models/Model.h"

namespace neml2
{
// input -> output identity map
template <typename T>
class IdentityMap : public Model
{
public:
  IdentityMap(const std::string & name,
              const std::string & from_axis_name,
              const std::string & from_var_name,
              const std::string & to_axis_name,
              const std::string & to_var_name);

  const LabeledAxisAccessor from;
  const LabeledAxisAccessor to;

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
} // namespace neml2
