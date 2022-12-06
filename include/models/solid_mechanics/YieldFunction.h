#pragma once

#include "models/Model.h"

/// Parent class for all yield functions
class YieldFunction : public Model
{
public:
  /// Calculate yield function knowing the corresponding hardening model
  YieldFunction(const std::string & name);

  /// The derivative and optionally the second derivative of the yield function w.r.t. the inputs
  virtual void set_dvalue(LabeledVector in,
                          LabeledMatrix dout_din,
                          LabeledTensor<1, 3> * d2out_din2 = nullptr) const = 0;

  /// Convenient shortcut to construct and return the model's second derivative
  virtual LabeledTensor<1, 3> d2value(LabeledVector in) const final;
};
