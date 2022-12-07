#pragma once

#include "models/Model.h"

template <bool rate>
class LinearIsotropicElasticity : public Model
{
public:
  LinearIsotropicElasticity(const std::string & name, Scalar E, Scalar nu);

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  Scalar _E;
  Scalar _nu;
};
