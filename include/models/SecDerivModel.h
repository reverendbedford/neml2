#pragma once

#include "models/Model.h"

/// Similar to `Model`, but provides an additional method `set_dvalue` that should be implemented.
class SecDerivModel : public Model
{
public:
  using Model::Model;

  /// Convenient shortcut to construct and return the model's first derivative
  virtual LabeledMatrix dvalue(LabeledVector in) const;

  /// Convenient shortcut to construct and return the model's second derivative
  virtual LabeledTensor<1, 3> d2value(LabeledVector in) const;

  /// Convenient shortcut to construct and return the model's first and second derivative
  virtual std::tuple<LabeledMatrix, LabeledTensor<1, 3>> dvalue_and_d2value(LabeledVector in) const;

protected:
  /// The derivative and optionally the second derivative of the yield function w.r.t. the inputs
  virtual void set_dvalue(LabeledVector in,
                          LabeledMatrix dout_din,
                          LabeledTensor<1, 3> * d2out_din2 = nullptr) const = 0;
};
