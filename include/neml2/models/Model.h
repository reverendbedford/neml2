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

#include "neml2/tensors/LabeledVector.h"
#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/tensors/LabeledTensor3D.h"
#include "neml2/models/LabeledAxisInterface.h"
#include "neml2/base/Registry.h"
#include "neml2/base/NEML2Object.h"
#include "neml2/base/Factory.h"
#include "neml2/solvers/NonlinearSystem.h"

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

  /// Whether this model is implicit
  virtual bool implicit() const { return false; }

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

  /// Tell this model which parameters require grad
  void trace_parameters(const std::map<std::string, bool> &);

  /// Set this model's parameters
  void set_parameters(const std::map<std::string, torch::Tensor> &);

  /// Get a parameter's value
  template <typename T>
  T get_parameter(const std::string & name, bool recurse = false) const
  {
    return T(named_parameters(recurse)[name]);
  }

  /**
   * @brief Use automatic differentiation to calculate the derivatives w.r.t. to the parameter
   *
   * If the parameter is batched, the batch shape must be the _same_ with the batch shape of the
   * output \p o.
   *
   * @param o The `BatchTensor` to to be differentiated
   * @param p The parameter to take derivatives with respect to
   * @return BatchTensor $\partial o/\partial p$
   */
  BatchTensor dparam(const BatchTensor & o, const BatchTensor & p) const;

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

  const std::vector<Model *> & registered_models() const { return _registered_models; }

  const std::set<LabeledAxisAccessor> & consumed_variables() const { return _consumed_vars; }
  const std::set<LabeledAxisAccessor> & provided_variables() const { return _provided_vars; }
  const std::set<LabeledAxisAccessor> & additional_outputs() const { return _additional_outputs; }

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

  /// Declare an input variable
  template <typename T>
  LabeledAxisAccessor declare_input_variable(const std::vector<std::string> & names)
  {
    auto accessor = declare_variable<T>(_input, names);
    _consumed_vars.insert(accessor);
    return accessor;
  }

  /// Declare an input variable on a subaxis
  template <typename T>
  LabeledAxisAccessor declare_input_variable(const std::string & subaxis,
                                             const std::vector<std::string> & names)
  {
    auto new_names = names;
    new_names.insert(new_names.begin(), subaxis);
    return declare_input_variable<T>(new_names);
  }

  /// Declare an input variable with known storage size
  LabeledAxisAccessor declare_input_variable(TorchSize sz, const std::vector<std::string> & names)
  {
    auto accessor = declare_variable(_input, sz, names);
    _consumed_vars.insert(accessor);
    return accessor;
  }

  /// Declare an output variable
  template <typename T>
  LabeledAxisAccessor declare_output_variable(const std::vector<std::string> & names)
  {
    auto accessor = declare_variable<T>(_output, names);
    _provided_vars.insert(accessor);
    return accessor;
  }

  /// Declare an output variable on a subaxis
  template <typename T>
  LabeledAxisAccessor declare_output_variable(const std::string & subaxis,
                                              const std::vector<std::string> & names)
  {
    auto new_names = names;
    new_names.insert(new_names.begin(), subaxis);
    return declare_output_variable<T>(new_names);
  }

  /// Declare an output variable with known storage size
  LabeledAxisAccessor declare_output_variable(TorchSize sz, const std::vector<std::string> & names)
  {
    auto accessor = declare_variable(_output, sz, names);
    _provided_vars.insert(accessor);
    return accessor;
  }

  virtual void setup() { setup_layout(); }

  /**
  Register a model that the current model may use during its evaluation. No dependency information
  is added.

  NOTE: We also register this model as a submodule (in torch's language), so that when *this*
  `Model` is sent to another device, the registered `Model` is also sent to that device.
  */
  void register_model(std::shared_ptr<Model> model, bool merge_input = true);

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
