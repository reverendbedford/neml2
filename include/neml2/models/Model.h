// Copyright 2023, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "neml2/base/Registry.h"
#include "neml2/base/Factory.h"
#include "neml2/base/UniqueVector.h"
#include "neml2/base/NEML2Object.h"
#include "neml2/models/LabeledAxisInterface.h"
#include "neml2/solvers/NonlinearSystem.h"

#include "neml2/tensors/LabeledVector.h"
#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/tensors/LabeledTensor3D.h"

namespace neml2
{
/**
 * @brief The base class for all constitutive models.
 *
 * A model maps some input to output. The forward operator (and its derivative) is defined in the
 * method \p set_value. All concrete models must provide the implementation of the forward operator
 * by overriding the \p set_value method.
 */
class Model : public NEML2Object, public LabeledAxisInterface, public NonlinearSystem
{
public:
  static OptionSet expected_options();

  /**
   * @brief Construct a new Model object
   *
   * @param options The options extracted from the input file
   */
  Model(const OptionSet & options);

  /**
   * @brief Recursively send this model and its sub-models to the target device.
   *
   * What this does behind the scenes is just sending all the model parameters to the target device.
   * This operation is recursively applied on of the sub-models.
   *
   * @param device The target device
   */
  void to(const torch::Device & device);

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

  /// Whether this model is implicit
  virtual bool implicit() const { return false; }

  /// The models that may be used during the evaluation of this model
  const std::vector<Model *> & registered_models() const { return _registered_models; }

  /// The variables that this model depends on
  const std::set<LabeledAxisAccessor> & consumed_variables() const { return _consumed_vars; }

  /// The variables that this model defines as part of its output
  const std::set<LabeledAxisAccessor> & provided_variables() const { return _provided_vars; }

  /**
   * The additional variables that this model should provide. Typically these variables are not
   * directly computed by this model, instead they come from other information that this model
   * _knows_, e.g., directly from the input variables.
   */
  const std::set<LabeledAxisAccessor> & additional_outputs() const { return _additional_outputs; }

  /**
   * Validate the currently requested AD settings.
   *  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   *  AD_1st_deriv   AD_2nd_deriv   comment
   *          true           true   okay, just slow
   *          true          false   error, this is a weird case
   *         false           true   okay
   *         false          false   great, everything handcoded
   *  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   */
  void check_AD_limitation() const;

  /// Tell this model to use AD to get derivatives
  void use_AD_derivatives(bool first = true, bool second = true);

  /// Convenient shortcut to construct and return the model value
  virtual LabeledVector value(const LabeledVector & in) const;

  /// Convenient shortcut to construct and return the model derivative
  virtual LabeledMatrix dvalue(const LabeledVector & in) const;

  /// Convenient shortcut to construct and return the model's second derivative
  virtual LabeledTensor3D d2value(const LabeledVector & in) const;

  /// Convenient shortcut to construct and return the model value and its derivative
  virtual std::tuple<LabeledVector, LabeledMatrix> value_and_dvalue(const LabeledVector & in) const;

  /// Convenient shortcut to construct and return the model's first and second derivative
  virtual std::tuple<LabeledMatrix, LabeledTensor3D>
  dvalue_and_d2value(const LabeledVector & in) const;

  /// Convenient shortcut to construct and return the model's value, first and second derivative
  virtual std::tuple<LabeledVector, LabeledMatrix, LabeledTensor3D>
  value_and_dvalue_and_d2value(const LabeledVector & in) const;

  bool has_parameter(const std::string & name) const { return _param_ids.count(name); }

  bool has_nonlinear_parameter(const std::string & name) const { return _nl_params.count(name); }

  /**
   * @brief (Recursively) get the named model parameters
   *
   * If \p recurse is set true, then each sub-model's parameters are prepended by the model name
   * followed by a dot ".". This is consistent with torch::nn::Module's naming convention.
   *
   * @param recurse Whether to recursively retrieve parameter names of sub-models.
   * @return A map from parameter name to parameter value
   */
  std::map<std::string, BatchTensor> named_parameters(bool recurse = false) const;

  /// Get a parameter's value
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & get_parameter(const std::string & name) const;

  /// Get a buffer's value
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & get_buffer(const std::string & name) const;

  /**
   * During the SOLVING stage, we update the state with \emph fixed forces, old forces, and old
   * state. This function caches those fixed values.
   */
  void cache_input(const LabeledVector & in);

  /**
   * A model can be treated as an implicit model. An implicit model need to be "solved": the state
   * variables should be iteratively updated until the residual becomes zero. During the SOLVING
   * stage, we only need the derivative of output with respect to the input state. During the
   * UPDATING stage, we only need the derivative of output with respect to the input forces, old
   * forces, and old state. Therefore, the model can/should avoid unnecessary computations by
   * examining the current `stage`.
   */
  enum Stage
  {
    SOLVING,
    UPDATING
  };
  static Model::Stage stage;

protected:
  /// The map between input -> output, and optionally its derivatives
  virtual void set_value(const LabeledVector & in,
                         LabeledVector * out,
                         LabeledMatrix * dout_din = nullptr,
                         LabeledTensor3D * d2out_din2 = nullptr) const = 0;

  /// Get the accessor for a given nonlinear parameter
  const LabeledAxisAccessor & nl_param(const std::string & name) const
  {
    return _nl_params.at(name);
  }

  /// Declare an input variable
  template <typename T>
  LabeledAxisAccessor declare_input_variable(const LabeledAxisAccessor & var)
  {
    auto accessor = declare_variable<T>(_input, var);
    _consumed_vars.insert(accessor);
    return accessor;
  }

  /// Declare an input variable with known storage size
  LabeledAxisAccessor declare_input_variable(const LabeledAxisAccessor & var, TorchSize sz)
  {
    auto accessor = declare_variable(_input, var, sz);
    _consumed_vars.insert(accessor);
    return accessor;
  }

  /// Declare an output variable
  template <typename T>
  LabeledAxisAccessor declare_output_variable(const LabeledAxisAccessor & var)
  {
    auto accessor = declare_variable<T>(_output, var);
    _provided_vars.insert(accessor);
    return accessor;
  }

  /// Declare an output variable with known storage size
  LabeledAxisAccessor declare_output_variable(const LabeledAxisAccessor & var, TorchSize sz)
  {
    auto accessor = declare_variable(_output, var, sz);
    _provided_vars.insert(accessor);
    return accessor;
  }

  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & declare_parameter(const std::string & name, const T & rawval);

  /**
   * @brief Declare a model parameter.
   *
   * @tparam T Parameter type. See @ref primitive for supported types.
   * @param name Name of the model parameter.
   * @param input_option_name Name of the input option that defines the value of the model
   * parameter.
   * @return T The value of the registered model parameter.
   */
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & declare_parameter(const std::string & name, const std::string & input_option_name);

  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & declare_buffer(const std::string & name, const T & rawval);

  /**
   * @brief Declare a model buffer.
   *
   * @tparam T Buffer type. See @ref primitive for supported types.
   * @param name Name of the model buffer.
   * @param input_option_name Name of the input option that defines the value of the model
   * buffer.
   * @return T The value of the registered model buffer.
   */
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  const T & declare_buffer(const std::string & name, const std::string & input_option_name);

  virtual void setup() { setup_layout(); }

  /**
  Register a model that the current model may use during its evaluation. No dependency information
  is added.

  NOTE: We also register this model as a submodule (in torch's language), so that when *this*
  `Model` is sent to another device, the registered `Model` is also sent to that device.
  */
  void register_model(std::shared_ptr<Model> model, bool merge_input = true);

  /**
   * Both register a model and return a reference
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<Model, T>>>
  T & include_model(const std::string & name, bool merge_input = true)
  {
    std::shared_ptr<Model> model = Factory::get_object_ptr<Model>("Models", name);

    register_model(model, merge_input);

    return *(std::dynamic_pointer_cast<T>(model));
  }

  virtual void
  assemble(const BatchTensor & x, BatchTensor * r, BatchTensor * J = nullptr) const override;

  /// Models *this* model may use during its evaluation
  std::vector<Model *> _registered_models;

  std::set<LabeledAxisAccessor> _consumed_vars;
  std::set<LabeledAxisAccessor> _provided_vars;
  std::set<LabeledAxisAccessor> _additional_outputs;

private:
  class ParameterValueBase
  {
  public:
    virtual ~ParameterValueBase() = default;

    /**
     * String identifying the type of parameter stored.
     * Must be reimplemented in derived classes.
     */
    virtual std::string type() const = 0;

    /// Send the value to the target device
    virtual void to(const torch::Device &) = 0;

    /// Convert the parameter value to a BatchTensor
    virtual operator BatchTensor() const = 0;
  };

  /// Concrete definition of a parameter value for a specified type
  template <typename T>
  class ParameterValue : public ParameterValueBase
  {
  public:
    ParameterValue() = default;

    ParameterValue(const T & value)
      : _value(value)
    {
    }

    virtual std::string type() const override { return utils::demangle(typeid(T).name()); }

    virtual void to(const torch::Device & device) override { _value = _value.to(device); }

    virtual operator BatchTensor() const override { return BatchTensor(_value); }

    const T & get() const { return _value; }

    T & set() { return _value; }

  private:
    /// Stored option value
    T _value;
  };

  LabeledAxis & _input;
  LabeledAxis & _output;

  std::map<std::string, size_t> _param_ids;
  std::vector<std::string> _param_names;
  UniqueVector<ParameterValueBase> _param_values;

  std::map<std::string, size_t> _buffer_ids;
  std::vector<std::string> _buffer_names;
  UniqueVector<ParameterValueBase> _buffer_values;

  std::map<std::string, LabeledAxisAccessor> _nl_params;

  bool _AD_1st_deriv;
  bool _AD_2nd_deriv;

  /// Cached input while solving this implicit model
  LabeledVector _cached_in;
};

template <typename T, typename>
const T &
Model::get_parameter(const std::string & name) const
{
  auto id = _param_ids.at(name);
  const auto & base_prop = _param_values[id];
  const auto prop = dynamic_cast<const ParameterValue<T> *>(&base_prop);
  neml_assert_dbg(prop, "Internal error, parameter cast failure.");
  return prop->get();
}

template <typename T, typename>
const T &
Model::get_buffer(const std::string & name) const
{
  auto id = _buffer_ids.at(name);
  const auto & base_prop = _buffer_values[id];
  const auto prop = dynamic_cast<const ParameterValue<T> *>(&base_prop);
  neml_assert_dbg(prop, "Internal error, buffer cast failure.");
  return prop->get();
}

template <typename T, typename>
const T &
Model::declare_parameter(const std::string & name, const T & rawval)
{
  neml_assert(std::find(_param_names.begin(), _param_names.end(), name) == _param_names.end(),
              "Trying to declare a parameter named ",
              name,
              " that already exists.");

  auto val = std::make_unique<ParameterValue<T>>(rawval);
  _param_ids.emplace(name, _param_ids.size());
  _param_names.push_back(name);
  auto & base_prop = _param_values.add_pointer(std::move(val));
  auto prop = dynamic_cast<ParameterValue<T> *>(&base_prop);
  neml_assert(prop, "Internal error, parameter cast failure.");
  return prop->get();
}

template <typename T, typename>
const T &
Model::declare_buffer(const std::string & name, const T & rawval)
{
  neml_assert(std::find(_buffer_names.begin(), _buffer_names.end(), name) == _buffer_names.end(),
              "Trying to declare a buffer named ",
              name,
              " that already exists.");

  auto val = std::make_unique<ParameterValue<T>>(rawval);
  _buffer_ids.emplace(name, _buffer_ids.size());
  _buffer_names.push_back(name);
  auto & base_prop = _buffer_values.add_pointer(std::move(val));
  auto prop = dynamic_cast<ParameterValue<T> *>(&base_prop);
  neml_assert(prop, "Internal error, parameter cast failure.");
  return prop->get();
}

template <typename T, typename>
const T &
Model::declare_buffer(const std::string & name, const std::string & input_option_name)
{
  if (options().contains<T>(input_option_name))
    return declare_buffer(name, options().get<T>(input_option_name));
  else if (options().contains<CrossRef<T>>(input_option_name))
    return declare_buffer(name, T(options().get<CrossRef<T>>(input_option_name)));

  throw NEMLException(
      "Trying to register buffer named " + name + " from input option named " + input_option_name +
      " of type " + utils::demangle(typeid(T).name()) +
      ". Make sure you provided the correct buffer name, option name, and buffer type. Note that "
      "the buffer type can either be a plain type, a cross-reference, or an interpolator.");
}

} // namespace neml2
