#pragma once

#include "models/Model.h"

/// Similar to `Model`, but uses automatic differention to get the model derivative.
class ADModel : public Model
{
public:
  using Model::Model;

  virtual LabeledMatrix dvalue(LabeledVector in) const;
};
