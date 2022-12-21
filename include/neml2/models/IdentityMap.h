#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
// input -> output identity map
template <typename T>
class IdentityMap : public Model
{
public:
  IdentityMap(const std::string & name,
              const std::vector<std::string> & from_var,
              const std::vector<std::string> & to_var);

  const LabeledAxisAccessor from;
  const LabeledAxisAccessor to;

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
} // namespace neml2
