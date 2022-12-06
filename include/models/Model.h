#pragma once

#include "tensors/LabeledVector.h"
#include "tensors/LabeledMatrix.h"
#include "models/LabeledAxisInterface.h"

/// Class that maps some input -> output,
/// which is also the broader definition of a constitutive model :)
class Model : public LabeledAxisInterface
{
public:
  Model(const std::string & name);

  /// Get the name of the model
  std::string name() const { return _name; }

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

  /// The map between input -> output, and optionally its derivatives
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const = 0;

  /// Convenient shortcut to construct and return the model value
  virtual LabeledVector value(LabeledVector in) const;

  /// Convenient shortcut to construct and return the model derivative
  virtual LabeledMatrix dvalue(LabeledVector in) const;

  const std::vector<Model *> & registered_models() const { return _registered_models; }

protected:
  virtual void setup() { setup_layout(); }

  /// Register a model that the current model may use during its evaluation
  /// No dependency information is added
  template <class T>
  T & registerModel(Model & model)
  {
    input().merge(model.input());
    _registered_models.push_back(&model);
    return dynamic_cast<T &>(model);
  }

  /// Models *this* model may use during its evaluation
  std::vector<Model *> _registered_models;

private:
  std::string _name;
  LabeledAxis & _input;
  LabeledAxis & _output;
};
