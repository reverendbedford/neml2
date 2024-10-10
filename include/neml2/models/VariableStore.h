// Copyright 2024, UChicago Argonne, LLC
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

#include "neml2/base/NEML2Object.h"
#include "neml2/base/Storage.h"
#include "neml2/tensors/Variable.h"
#include "neml2/tensors/LabeledVector.h"
#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/tensors/LabeledTensor3D.h"

namespace neml2
{
// Foward decl
class Model;

class VariableStore
{
public:
  VariableStore(const OptionSet & options);

  LabeledAxis & declare_axis(const std::string & name);

  /// Setup the layouts of all the registered axes
  virtual void setup_layout();

  ///@{
  /// Definition of the input axis showing the layout of input variables
  LabeledAxis & input_axis() { return _input_axis; }
  const LabeledAxis & input_axis() const { return _input_axis; }
  ///@}

  ///@{
  /// Definition of the output axis showing the layout of output variables
  LabeledAxis & output_axis() { return _output_axis; }
  const LabeledAxis & output_axis() const { return _output_axis; }
  ///@}

  ///@{
  /// Variables
  Storage<VariableName, VariableBase> & variables() { return _variables; }
  const Storage<VariableName, VariableBase> & variables() const { return _variables; }
  ///@}

  /// Get the variable type of an input variable
  TensorType input_type(const VariableName &) const;
  /// Get the variable type of an output variable
  TensorType output_type(const VariableName &) const;

protected:
  /// Tell each input variable view which tensor storage(s) to view into
  virtual void setup_input_views(VariableStore * host = nullptr);

  /// Tell each output variable view which tensor storage(s) to view into
  virtual void setup_output_views();

  /// Declare an input variable
  template <typename T, typename... S>
  const Variable<T> & declare_input_variable(S &&... name)
  {
    const auto var_name = variable_name(std::forward<S>(name)...);
    declare_variable<T>(_input_axis, var_name);
    return *create_variable_view<T>(var_name);
  }

  /// Declare an input variable (with unknown base shape at compile time)
  template <typename... S>
  const Variable<Tensor> & declare_input_variable(Size sz, TensorType t, S &&... name)
  {
    const auto var_name = variable_name(std::forward<S>(name)...);
    declare_variable(_input_axis, var_name, sz);
    return *create_variable_view<Tensor>(var_name, t, sz);
  }

  /// Declare an input variable that is a list of tensors of fixed size
  template <typename T, typename... S>
  const Variable<Tensor> & declare_input_variable_list(Size list_size, S &&... name)
  {
    return declare_input_variable(
        list_size * T::const_base_storage, TensorType::kTensor, std::forward<S>(name)...);
  }

  /// Declare an output variable
  template <typename T, typename... S>
  Variable<T> & declare_output_variable(S &&... name)
  {
    const auto var_name = variable_name(std::forward<S>(name)...);
    declare_variable<T>(_output_axis, var_name);
    return *create_variable_view<T>(var_name);
  }

  /// Declare an input variable (with unknown base shape at compile time)
  template <typename... S>
  Variable<Tensor> & declare_output_variable(Size sz, TensorType t, S &&... name)
  {
    const auto var_name = variable_name(std::forward<S>(name)...);
    declare_variable(_output_axis, var_name, sz);
    return *create_variable_view<Tensor>(var_name, t, sz);
  }

  /// Declare an output variable that is a list of tensors of fixed size
  template <typename T, typename... S>
  Variable<Tensor> & declare_output_variable_list(Size list_size, S &&... name)
  {
    return declare_output_variable(
        list_size * T::const_base_storage, TensorType::kTensor, std::forward<S>(name)...);
  }

  /// Declare an item recursively on an axis
  template <typename T>
  void declare_variable(LabeledAxis & axis, const VariableName & var) const
  {
    declare_variable(axis, var, T::const_base_storage);
  }

  /// Declare an item (with known storage size) recursively on an axis
  void declare_variable(LabeledAxis & axis, const VariableName & var, Size sz) const
  {
    axis.add(var, sz);
  }

  /// Declare a subaxis recursively on an axis
  VariableName declare_subaxis(LabeledAxis & axis, const VariableName & subaxis) const
  {
    axis.add<LabeledAxis>(subaxis);
    return subaxis;
  }

private:
  // Helper method to construct variable name in place
  template <typename... S>
  VariableName variable_name(S &&... name) const
  {
    using FirstType = std::tuple_element_t<0, std::tuple<S...>>;

    if constexpr (sizeof...(name) == 1 && std::is_convertible_v<FirstType, std::string>)
    {
      if (_options.contains<VariableName>(name...))
        return _options.get<VariableName>(name...);
      return VariableName(std::forward<S>(name)...);
    }
    else
      return VariableName(std::forward<S>(name)...);
  }

  // Create a variable view (doesn't setup the view)
  template <typename T>
  Variable<T> * create_variable_view(const VariableName & name,
                                     TensorType t = TensorTypeEnum<T>::value,
                                     Size sz = -1)
  {
    if constexpr (std::is_same_v<T, Tensor>)
      neml_assert(sz > 0, "Creating a Variable<Tensor> requires a known storage size.");

    // Make sure we don't duplicate variables
    VariableBase * var_base_ptr = views.query_value(name);
    neml_assert(!var_base_ptr,
                name(),
                ": Trying to create variable ",
                name,
                ", but a variable with the same name already exists.");

    // Allocate
    if constexpr (std::is_same_v<T, Tensor>)
    {
      auto var = std::make_unique<Variable<Tensor>>(name, this, sz, t);
      var_base_ptr = views.set_pointer(name, std::move(var));
    }
    else
    {
      (void)t;
      auto var = std::make_unique<Variable<T>>(name, this);
      var_base_ptr = views.set_pointer(name, std::move(var));
    }

    // Cast it to the concrete type
    auto var_ptr = dynamic_cast<Variable<T> *>(var_base_ptr);
    neml_assert(
        var_ptr, "Internal error: Failed to cast variable ", name, " to its concrete type.");

    return var_ptr;
  }

  /**
   * @brief Parsed input file options. These options are useful for example when we declare a
   * variable using an input option name.
   *
   */
  const OptionSet _options;

  /// All the declared axes
  Storage<std::string, LabeledAxis> _axes;

  /// The input axis
  LabeledAxis & _input_axis;

  /// The output axis
  LabeledAxis & _output_axis;

  /// Variables
  Storage<VariableName, VariableBase> _variables;

  /// Tensor storage associated with the variables
  // Note: These tensors are only allocated in the host model
  ///@{
  /// Flattened variable values
  std::vector<Tensor> _variable_values_raw;
  /// Un-flattened variable values
  std::vector<Tensor> _variable_values;
  ///@}
};
} // namespace neml2
