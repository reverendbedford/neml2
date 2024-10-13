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

#include "neml2/tensors/LabeledAxisAccessor.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/LabeledVector.h"
#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/tensors/LabeledTensor3D.h"

namespace neml2
{
using VariableName = LabeledAxisAccessor;

// Forward declarations
class Model;

class VariableBase
{
public:
  VariableBase(const VariableName & name_in, const Model * owner, FType ftype, TensorType type);

  virtual ~VariableBase() = default;

  /// Name of this variable
  const VariableName & name() const { return _name; }

  /// The Model who declared this variable
  const Model & owner() const { return *_owner; }

  /// Variable ftype (role in a function definition)
  FType ftype() const { return _ftype; }

  /// Variable tensor type
  TensorType type() const { return _type; }

  /// @name Subaxis
  ///@{
  bool is_state() const;
  bool is_old_state() const;
  bool is_force() const;
  bool is_old_force() const;
  bool is_residual() const;
  bool is_parameter() const;
  bool is_solve_dependent() const;
  /// Check if the derivative with respect to this variable should be evaluated
  // Note that the check depends on whether we are currently solving nonlinear system
  bool is_dependent() const;
  ///@}

  /// Base shape of the variable
  virtual TensorShapeRef base_sizes() const = 0;

  /// Base storage of the variable
  Size base_storage() const { return utils::storage_size(base_sizes()); }

  /// Clone this variable
  virtual std::unique_ptr<VariableBase> clone() const = 0;

  /// Reference another variable
  virtual void ref(const VariableBase & other) = 0;

  /// Set the variable value (handles reshaping)
  virtual void set(const Tensor & val) = 0;

  /// Derivative
  Tensor & d(const VariableBase & var) { return _derivs[&var]; }

  /// Second derivative
  Tensor & d(const VariableBase & var1, const VariableBase & var2)
  {
    return _sec_derivs[&var1][&var2];
  }

protected:
  /// Name of the variable
  const VariableName _name;

  /// The model which declared this variable
  const Model * const _owner;

  /// Variable ftype (role in a function definition)
  const FType _ftype;

  /// Variable tensor type
  const TensorType _type;

private:
  /// Derivatives of this variable with respect to other variables
  std::map<const VariableBase *, Tensor> _derivs;

  /// Second derivatives of this variable with respect to other variables
  std::map<const VariableBase *, std::map<const VariableBase *, Tensor>> _sec_derivs;
};

/**
 * @brief Concrete definition of a variable
 *
 */
template <typename T>
class Variable : public VariableBase
{
public:
  template <typename T2 = T, typename = typename std::enable_if_t<!std::is_same_v<Tensor, T2>>>
  Variable(const VariableName & name_in,
           const Model * owner,
           FType ftype,
           TensorType type = TensorTypeEnum<T2>::value)
    : VariableBase(name_in, owner, ftype, type),
      _base_sizes(T::const_base_sizes),
      _value_ptr(nullptr)
  {
  }

  template <typename T2 = T, typename = typename std::enable_if_t<std::is_same_v<Tensor, T2>>>
  Variable(const VariableName & name_in,
           const Model * owner,
           TensorShapeRef base_shape,
           FType ftype,
           TensorType type = TensorType::kTensor)
    : VariableBase(name_in, owner, ftype, type),
      _base_sizes(base_shape),
      _value_ptr(nullptr)
  {
  }

  TensorShapeRef base_sizes() const override { return _base_sizes; }

  std::unique_ptr<VariableBase> clone() const override
  {
    if constexpr (std::is_same_v<T, Tensor>)
      return std::move(
          std::make_unique<Variable<T>>(name(), &owner(), base_sizes(), ftype(), type()));
    else
      return std::move(std::make_unique<Variable<T>>(name(), &owner(), ftype(), type()));
  }

  void ref(const VariableBase & var) override
  {
    const auto * var_ptr = dynamic_cast<const Variable<T> *>(&var);
    neml_assert(var_ptr,
                "Variable ",
                name(),
                " of type ",
                type(),
                " failed to reference another variable named ",
                var.name(),
                " of type ",
                var.type(),
                ": Dynamic cast failure.");
    _value_ptr = &var_ptr->value();
  }

  void set(const Tensor & val) override
  {
    neml_assert_dbg(val.base_storage() == base_storage(),
                    "Failed to set value for variable ",
                    name(),
                    " of type ",
                    type(),
                    ": Expected flattened base storage ",
                    base_storage(),
                    ", got ",
                    val.base_storage());
    (*this) = T(val.base_reshape(base_sizes()));
  }

  /// Suppressed constructor to prevent accidental dereferencing
  [[deprecated("Variable<T> must be assigned to references -- missing &")]] Variable(
      const Variable<T> &)
  {
  }

  /// Suppressed assignment operator to prevent accidental dereferencing
  [[deprecated("Variable<T> must be assigned to references -- missing &")]] void
  operator=(const Variable<T> &)
  {
  }

  /// Set the variable value
  void operator=(const T & val)
  {
    neml_assert_dbg(!_value_ptr,
                    "Failed to set value for variable ",
                    name(),
                    " of type ",
                    type(),
                    ": Variable is a reference.");
    _value = val;
  }

  /// Variable value
  const T & value() const { return _value_ptr ? *_value_ptr : _value; }

  /// Negation
  T operator-() const { return -value(); }

  operator T() const { return value(); }

  template <typename T2 = T, typename = typename std::enable_if_t<!std::is_same_v<T2, Tensor>>>
  operator Tensor() const
  {
    return value();
  }

protected:
  /// Base shape of the variable
  const TensorShape _base_sizes;

  /// Variable value (undefined if this is a referencing variable)
  T _value;

  /// Variable value (nullptr if this is a storing variable)
  const T * _value_ptr;
};

// Everything below is just for convenience: We just forward operations to the the variable values
// so that we can do
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//   var4 = (var1 - var2) * var3
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// instead of the (ugly?) expression below
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//   var4 = (var1.v - var2.v) * var3.v
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#define FWD_VARIABLE_BINARY_OP(op)                                                                 \
  template <typename T1,                                                                           \
            typename T2,                                                                           \
            typename = typename std::enable_if_t<std::is_base_of_v<VariableBase, T1> ||            \
                                                 std::is_base_of_v<VariableBase, T2>>>             \
  auto op(const T1 & a, const T2 & b)                                                              \
  {                                                                                                \
    if constexpr (std::is_base_of_v<VariableBase, T1> && std::is_base_of_v<VariableBase, T2>)      \
      return op(a.value(), b.value());                                                             \
                                                                                                   \
    if constexpr (std::is_base_of_v<VariableBase, T1> && !std::is_base_of_v<VariableBase, T2>)     \
      return op(a.value(), b);                                                                     \
                                                                                                   \
    if constexpr (!std::is_base_of_v<VariableBase, T1> && std::is_base_of_v<VariableBase, T2>)     \
      return op(a, b.value());                                                                     \
  }                                                                                                \
  static_assert(true)
FWD_VARIABLE_BINARY_OP(operator+);
FWD_VARIABLE_BINARY_OP(operator-);
FWD_VARIABLE_BINARY_OP(operator*);
FWD_VARIABLE_BINARY_OP(operator/);
}
