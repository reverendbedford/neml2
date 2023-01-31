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
#include "neml2/models/LabeledAxisInterface.h"

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

  const std::set<LabeledAxisAccessor> & consumed_variables() const { return _consumed_vars; }
  const std::set<LabeledAxisAccessor> & provided_variables() const { return _provided_vars; }

protected:
  /// The map between input -> output, and optionally its derivatives
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const = 0;

  /// Declare an input variable
  template <typename T>
  [[nodiscard]] LabeledAxisAccessor declareInputVariable(const std::vector<std::string> & names)
  {
    auto accessor = declareVariable<T>(_input, names);
    _consumed_vars.insert(accessor);
    return accessor;
  }

  /// Declare an input variable with known storage size
  [[nodiscard]] LabeledAxisAccessor declareInputVariable(TorchSize sz,
                                                         const std::vector<std::string> & names)
  {
    auto accessor = declareVariable(_input, sz, names);
    _consumed_vars.insert(accessor);
    return accessor;
  }

  /// Declare an output variable
  template <typename T>
  [[nodiscard]] LabeledAxisAccessor declareOutputVariable(const std::vector<std::string> & names)
  {
    auto accessor = declareVariable<T>(_output, names);
    _provided_vars.insert(accessor);
    return accessor;
  }

  /// Declare an output variable with known storage size
  template <typename... S>
  [[nodiscard]] LabeledAxisAccessor declareOutputVariable(TorchSize sz,
                                                          const std::vector<std::string> & names)
  {
    auto accessor = declareVariable(_output, sz, names);
    _provided_vars.insert(accessor);
    return accessor;
  }

  virtual void setup() { setup_layout(); }

  /**
  Register a model that the current model may use during its evaluation. No dependency information
  is added.

  NOTE: We also register this model as a submodule (in torch's language), so that when *this*
  `Model` is send to another device, the registered `Model` is also sent to that device.
  */
  void register_model(std::shared_ptr<Model> model, bool merge_input = true);

  /// Models *this* model may use during its evaluation
  std::vector<std::shared_ptr<Model>> _registered_models;

  std::set<LabeledAxisAccessor> _consumed_vars;
  std::set<LabeledAxisAccessor> _provided_vars;

private:
  LabeledAxis & _input;
  LabeledAxis & _output;
};
} // namespace neml2
