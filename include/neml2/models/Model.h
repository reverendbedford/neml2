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

#include "neml2/models/Data.h"
#include "neml2/models/ParameterStore.h"
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
class Model : public Data,
              public ParameterStore,
              public LabeledAxisInterface,
              public NonlinearSystem
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
  virtual void to(const torch::Device & device) override;

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

  /**
   * @brief Get the named parameters
   *
   * @param recurse Whether to recursively retrieve parameters from sub-models.
   * @return A map from parameter name to parameter value
   */
  virtual std::map<std::string, BatchTensor> named_parameters(bool recurse) const;
  using ParameterStore::named_parameters;

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

  /// Declaration of nonlinear parameters may require manipulation of input
  friend class ParameterStore;

protected:
  /// The map between input -> output, and optionally its derivatives
  virtual void set_value(const LabeledVector & in,
                         LabeledVector * out,
                         LabeledMatrix * dout_din = nullptr,
                         LabeledTensor3D * d2out_din2 = nullptr) const = 0;

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

  virtual void setup() { setup_layout(); }

  /**
   * @brief Register a model that the current model may use during its evaluation. No dependency
   * information is added.
   *
   * @param model The model to register
   * @param merge_input Whether to merge the input of the registered model into *this* model's
   * input.
   */
  void register_model(std::shared_ptr<Model> model, bool merge_input = true);

  /**
   * Both register a model and return a reference
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<Model, T>>>
  T & include_model(const std::string & name, bool merge_input = true)
  {
    auto model = Factory::get_object_ptr<Model>("Models", name);
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
  LabeledAxis & _input;
  LabeledAxis & _output;

  bool _AD_1st_deriv;
  bool _AD_2nd_deriv;

  /// Cached input while solving this implicit model
  LabeledVector _cached_in;
};
} // namespace neml2
