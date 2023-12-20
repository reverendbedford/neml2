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

#include "neml2/tensors/LabeledAxisAccessor.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/LabeledVector.h"
#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/tensors/LabeledTensor3D.h"

namespace neml2
{
// Forward declarations
class Derivative;
class SecondDerivative;

class VariableBase
{
public:
  VariableBase(const LabeledAxisAccessor & name_in)
    : _name(name_in),
      _value_storage(nullptr),
      _derivative_storage(nullptr),
      _second_derivative_storage(nullptr)
  {
  }

  virtual ~VariableBase() = default;

  /// Cache the variable's batch shape
  virtual void cache(TorchShapeRef batch_shape);

  /// Setup the variable's views into blocks of the storage
  virtual void setup_views(const LabeledVector * value,
                           const LabeledMatrix * deriv = nullptr,
                           const LabeledTensor3D * secderiv = nullptr);

  /// Setup the variable's views into blocks of the storage
  virtual void setup_views(const VariableBase * other);

  /// Is this variable a leaf variable?
  bool is_leaf() const { return _args.empty(); }

  /// Arguments
  const std::vector<LabeledAxisAccessor> & args() const { return _args; }

  /// Add an argument
  void add_arg(const VariableBase & arg) { _args.push_back(arg.name()); }

  /// Clear arguments
  void clear_args() { _args.clear(); }

  /// Create a wrapper representing the derivative dy/dx
  Derivative d(const VariableBase & x);

  /// Create a wrapper representing the second derivative d2y/dx2
  Derivative d(const VariableBase & x1, const VariableBase & x2);

  /// Raw (flattened) variable value (view)
  BatchTensor raw_value() const { return _raw_value; }

  /// Name of this variable
  const LabeledAxisAccessor & name() const { return _name; }

  /// Batch shape
  TorchShapeRef batch_sizes() const { return _batch_sizes; }

  /// Base shape
  virtual TorchShapeRef base_sizes() const = 0;

  /// Batch dimension
  TorchSize batch_dim() const { return _batch_sizes.size(); }

  /// Base dimension
  TorchSize base_dim() const { return base_sizes().size(); }

  /// Base storage
  TorchSize base_storage() const { return utils::storage_size(base_sizes()); }

  /// Total shape
  virtual TorchShapeRef sizes() const = 0;

protected:
  /// Name of the variable
  const LabeledAxisAccessor _name;

  /// Batch shape of this variable
  TorchShape _batch_sizes;

  /// Names of the variables that this variable depends on
  std::vector<LabeledAxisAccessor> _args;

  /// The raw (flattened) variable value
  BatchTensor _raw_value;

  /// The derivative of this variable w.r.t. arguments.
  std::map<LabeledAxisAccessor, BatchTensor> _dvalue_d;

  /// The second derivative of this variable w.r.t. arguments.
  std::map<LabeledAxisAccessor, std::map<LabeledAxisAccessor, BatchTensor>> _d2value_d;

  /// The value storage that this variable is viewing into
  const LabeledVector * _value_storage;

  /// The derivative storage that this variable is viewing into
  const LabeledMatrix * _derivative_storage;

  /// The second derivative storage that this variable is viewing into
  const LabeledTensor3D * _second_derivative_storage;
};

/**
 * @brief Concrete definition of a variable
 *
 */
template <typename T>
class Variable : public VariableBase
{
public:
  template <typename T2 = T, typename = typename std::enable_if_t<!std::is_same_v<BatchTensor, T2>>>
  Variable(const LabeledAxisAccessor & name_in)
    : VariableBase(name_in),
      _base_sizes(T::const_base_sizes)
  {
  }

  template <typename T2 = T, typename = typename std::enable_if_t<std::is_same_v<BatchTensor, T2>>>
  Variable(const LabeledAxisAccessor & name_in, TorchShapeRef base_shape)
    : VariableBase(name_in),
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

  virtual TorchShapeRef base_sizes() const override { return _base_sizes; }

  virtual TorchShapeRef sizes() const override { return _sizes; }

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
  void operator=(const BatchTensor & val)
  {
    _value.index_put_({torch::indexing::Slice()},
                      val.batch_expand(batch_sizes()).base_reshape(base_sizes()));
  }

  /// The variable value with the correct logical shape
  const T & value() const { return _value; }

  /// Negation
  T operator-() const { return -_value; }

  operator T() const { return _value; }

  /// Set the batch shape and base shape according to \p val
  virtual void cache(TorchShapeRef batch_shape) override
  {
    VariableBase::cache(batch_shape);
    _sizes = utils::add_shapes(batch_shape, _base_sizes);
  }

  template <typename T2 = T, typename = typename std::enable_if_t<!std::is_same_v<T2, BatchTensor>>>
  operator BatchTensor() const
  {
    return _value;
  }

protected:
  /// Base shape of this variable
  const TorchShape _base_sizes;

  /// Shape of this variable
  TorchShape _sizes;

  /// Tensor value with the correct logical shape
  T _value;
};

class Derivative
{
public:
  Derivative(BatchTensor & val)
    : _value(val)
  {
  }

  const BatchTensor & value() const { return _value; }

  void operator=(const BatchTensor & val);

private:
  BatchTensor & _value;
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
