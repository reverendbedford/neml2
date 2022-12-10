#pragma once

#include "tensors/LabeledVector.h"
#include "tensors/LabeledMatrix.h"
#include "models/LabeledAxisInterface.h"

/**
Class that maps some input -> output, which is also the broader definition of constitutive model.
*/
class Model : public torch::nn::Module, public LabeledAxisInterface
{
public:
  Model(const std::string & name);

  /// Definition of the input variables
  /// @{
  LabeledAxis & input() { return _input; }
  const LabeledAxis & input() const { return _input; }
  /// @}

  /// Which variables this object defines as output
  /// @{
  LabeledAxis & output() { return _output; }
  const LabeledAxis & output() const { return _output; }
  /// @}

  /// Convenient shortcut to construct and return the model value
  virtual LabeledVector value(LabeledVector in) const;

  /// Convenient shortcut to construct and return the model derivative
  /// NOTE: this method is inefficient and not recommended for use.
  /// Consider using `value` or `value_and_dvalue` if possible.
  virtual LabeledMatrix dvalue(LabeledVector in) const;

  /// Convenient shortcut to construct and return the model value and its derivative
  virtual std::tuple<LabeledVector, LabeledMatrix> value_and_dvalue(LabeledVector in) const;

  const std::vector<std::shared_ptr<Model>> & registered_models() const
  {
    return _registered_models;
  }

protected:
  /// The map between input -> output, and optionally its derivatives
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const = 0;

  virtual void setup() { setup_layout(); }

  /**
  Register a model that the current model may use during its evaluation. No dependency information
  is added.

  NOTE: We also register this model as a submodule (in torch's language), so that when *this*
  `Model` is send to another device, the registered `Model` is also sent to that device.
  */
  void register_model(std::shared_ptr<Model> model);

  /// Models *this* model may use during its evaluation
  std::vector<std::shared_ptr<Model>> _registered_models;

private:
  LabeledAxis & _input;
  LabeledAxis & _output;
};
