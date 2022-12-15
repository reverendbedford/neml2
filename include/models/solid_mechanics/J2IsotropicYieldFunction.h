#pragma once

#include "models/solid_mechanics/IsotropicYieldFunction.h"

namespace neml2
{
class J2IsotropicYieldFunction : public IsotropicYieldFunction
{
public:
  J2IsotropicYieldFunction(const std::string & name);

protected:
  /// The value of the yield function
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  /// The derivative of the yield function w.r.t. hardening variables
  virtual void set_dvalue(LabeledVector in,
                          LabeledMatrix dout_din,
                          LabeledTensor<1, 3> * d2out_din2 = nullptr) const;
};
} // namespace neml2
