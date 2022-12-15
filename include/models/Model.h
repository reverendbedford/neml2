#pragma once

#include "tensors/LabeledVector.h"
#include "tensors/LabeledMatrix.h"
#include "models/LabeledAxisInterface.h"

namespace neml2
{
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

  const std::vector<LabeledAxisAccessor> & consumed_variables() const { return _consumed_vars; }
  const std::vector<LabeledAxisAccessor> & provided_variables() const { return _provided_vars; }

protected:
  /// The map between input -> output, and optionally its derivatives
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const = 0;

  /// Declare an input variable
  template <typename T,
            typename... S,
            typename = std::enable_if_t<are_all_convertible<std::string, S...>::value>>
  [[nodiscard]] LabeledAxisAccessor declareInputVariable(S &&... name)
  {
    auto accessor = declareVariable<T>(_input, std::forward<S>(name)...);
    _consumed_vars.push_back(accessor);
    return accessor;
  }

  /// Declare an input variable with known storage size
  template <typename... S>
  [[nodiscard]] LabeledAxisAccessor declareInputVariable(TorchSize sz, S &&... name)
  {
    auto accessor = declareVariable(_input, sz, std::forward<S>(name)...);
    _consumed_vars.push_back(accessor);
    return accessor;
  }

  /// Declare an output variable
  template <typename T,
            typename... S,
            typename = std::enable_if_t<are_all_convertible<std::string, S...>::value>>
  [[nodiscard]] LabeledAxisAccessor declareOutputVariable(S &&... name)
  {
    auto accessor = declareVariable<T>(_output, std::forward<S>(name)...);
    _provided_vars.push_back(accessor);
    return accessor;
  }

  /// Declare an output variable with known storage size
  template <typename... S>
  [[nodiscard]] LabeledAxisAccessor declareOutputVariable(TorchSize sz, S &&... name)
  {
    auto accessor = declareVariable(_output, sz, std::forward<S>(name)...);
    _provided_vars.push_back(accessor);
    return accessor;
  }

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

  std::vector<LabeledAxisAccessor> _consumed_vars;
  std::vector<LabeledAxisAccessor> _provided_vars;

private:
  LabeledAxis & _input;
  LabeledAxis & _output;
};
} // namespace neml2
