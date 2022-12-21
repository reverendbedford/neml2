#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
// input -> output identity map
template <typename T>
class SumModel : public Model
{
public:
  SumModel(const std::string & name,
           const std::vector<std::vector<std::string>> & from_vars,
           const std::vector<std::string> & to_var);

  const LabeledAxisAccessor to;
  std::vector<LabeledAxisAccessor> from;

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
} // namespace neml2
