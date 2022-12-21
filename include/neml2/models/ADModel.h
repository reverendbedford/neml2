#pragma once

#include "neml2/models/Model.h"

namespace neml2
{
/// Similar to `Model`, but uses automatic differention to get the model derivative.
class ADModel : public Model
{
public:
  using Model::Model;

  virtual std::tuple<LabeledVector, LabeledMatrix> value_and_dvalue(LabeledVector in) const;
};
} // namespace neml2
