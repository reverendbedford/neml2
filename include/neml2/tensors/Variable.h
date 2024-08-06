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
class Derivative;
class Model;

class VariableBase
{
public:
  VariableBase(const VariableName & name_in, const Model * owner);

  virtual ~VariableBase() = default;

  /// Cache the variable's batch shape
  virtual void cache(TensorShapeRef batch_shape);

  /// Setup the variable's views into blocks of the storage
  virtual void setup_views(const LabeledVector * value,
                           const LabeledMatrix * deriv = nullptr,
                           const LabeledTensor3D * secderiv = nullptr);

  /// Setup the variable's views following another variable
  virtual void setup_views(const VariableBase * other);

  /// Set requires_grad for the underlying storage
  virtual void requires_grad_(bool req = true) = 0;

  /// Create a wrapper representing the derivative dy/dx
  Derivative d(const VariableBase & x);

  /// Create a wrapper representing the second derivative d2y/dx2
  Derivative d(const VariableBase & x1, const VariableBase & x2);

  /// Raw flattened variable value
  const Tensor & raw_value() const { return _raw_value; }

  /// Variable value of the logical shape
  virtual const Tensor tensor() const = 0;

  /// Name of this variable
  const VariableName & name() const { return _name; }

  /// The owner of this variable
  const Model & owner() const { return *_owner; }

  /// The source variable
  const VariableBase * src() const { return _src; }

  /// Batch shape
  TensorShapeRef batch_sizes() const { return _batch_sizes; }

  /// Base shape
  virtual TensorShapeRef base_sizes() const = 0;

  /// Batch dimension
  Size batch_dim() const { return _batch_sizes.size(); }

  /// Base dimension
  Size base_dim() const { return base_sizes().size(); }

  /// Base storage
  Size base_storage() const { return utils::storage_size(base_sizes()); }

  /// Total shape
  virtual TensorShapeRef sizes() const = 0;

  /// Variable type
  virtual TensorType type() const = 0;

  /// @name Subaxis
  ///@{
  bool is_state() const { return _is_state; }
  bool is_old_state() const { return _is_old_state; }
  bool is_force() const { return _is_force; }
  bool is_old_force() const { return _is_old_force; }
  bool is_residual() const { return _is_residual; }
  bool is_parameter() const { return _is_parameter; }
  bool is_other() const { return _is_other; }
  bool is_solve_dependent() const { return _is_solve_dependent; }
  ///@}

  /// Check if the derivative with respect to this variable should be evaluated
  // Note that the check depends on whether we are currently solving nonlinear system
  bool is_dependent() const;

protected:
  /// Name of the variable
  const VariableName _name;

  /// The model which declared this variable
  const Model * _owner;

  /// Batch shape of this variable
  TensorShape _batch_sizes;

  /// The raw (flattened) variable value
  Tensor _raw_value;

  /// The derivative of this variable w.r.t. arguments.
  std::map<VariableName, Tensor> _dvalue_d;

  /// The second derivative of this variable w.r.t. arguments.
  std::map<VariableName, std::map<VariableName, Tensor>> _d2value_d;

  /// The source variable this variable follows
  const VariableBase * _src;

  /// @name subaxis
  ///@{
  const bool _is_state;
  const bool _is_old_state;
  const bool _is_force;
  const bool _is_old_force;
  const bool _is_residual;
  const bool _is_parameter;
  const bool _is_other;
  const bool _is_solve_dependent;
  ///@}
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
           TensorType type = TensorTypeEnum<T2>::value)
    : VariableBase(name_in, owner),
      _type(type),
      _base_sizes(T::const_base_sizes)
  {
  }

  template <typename T2 = T, typename = typename std::enable_if_t<std::is_same_v<Tensor, T2>>>
  Variable(const VariableName & name_in,
           const Model * owner,
           TensorShapeRef base_shape,
           TensorType type = TensorType::kTensor)
    : VariableBase(name_in, owner),
      _type(type),
      _base_sizes(base_shape.vec())
  {
  }

  virtual void setup_views(const LabeledVector * value,
                           const LabeledMatrix * deriv = nullptr,
                           const LabeledTensor3D * secderiv = nullptr) override
  {
    VariableBase::setup_views(value, deriv, secderiv);
    if (value)
      _value = T(_raw_value.view(sizes()), batch_dim());
  }

  virtual void setup_views(const VariableBase * other) override
  {
    VariableBase::setup_views(other);
    _value = T(_raw_value.view(sizes()), batch_dim());
  }

  virtual void requires_grad_(bool req = true) override { _value.requires_grad_(req); }

  virtual TensorShapeRef base_sizes() const override { return _base_sizes; }

  virtual TensorShapeRef sizes() const override { return _sizes; }

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

  /**
   * @brief Set the raw value to store \p val
   *
   * Note that this is an in-place operation, and so we must reshape (flatten base dimensions of) \p
   * val and modify raw_value.
   */
  void operator=(const Tensor & val)
  {
    _value.index_put_({torch::indexing::Slice()},
                      val.batch_expand(batch_sizes()).base_reshape(base_sizes()));
  }

  /// Variable value of the logical shape
  const T & value() const { return _value; }

  /// Variable value of the logical shape
  virtual const Tensor tensor() const override { return _value; }

  virtual TensorType type() const override { return _type; }

  /// Negation
  T operator-() const { return -_value; }

  operator T() const { return _value; }

  /// Set the batch shape and base shape according to \p val
  virtual void cache(TensorShapeRef batch_shape) override
  {
    VariableBase::cache(batch_shape);
    _sizes = utils::add_shapes(batch_shape, _base_sizes);
  }

  template <typename T2 = T, typename = typename std::enable_if_t<!std::is_same_v<T2, Tensor>>>
  operator Tensor() const
  {
    return _value;
  }

protected:
  /// Variable tensor type
  const TensorType _type;

  /// Base shape of this variable
  const TensorShape _base_sizes;

  /// Shape of this variable
  TensorShape _sizes;

  /// Variable value of the logical shape
  T _value;
};

class Derivative
{
public:
  Derivative(Tensor & val)
    : _value(val)
  {
  }

  const Tensor & value() const { return _value; }

  Derivative & operator=(const Tensor & val);

private:
  Tensor & _value;
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
