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
using VariableName = LabeledAxisAccessor;

// Forward declarations
class Derivative;

class VariableBase
{
public:
  VariableBase(const VariableName & name_in, const AssemblyMode & assembly_mode)
    : _name(name_in),
      _assembly_mode(assembly_mode),
      // AssemblyMode::INPLACE
      _value_storage(nullptr),
      _derivative_storage(nullptr),
      _second_derivative_storage(nullptr),
      // AssemblyMode::CONCATENATION
      __args_idx(nullptr),
      __raw_value(nullptr),
      __dvalue_d(nullptr),
      __d2value_d(nullptr)
  {
  }

  virtual ~VariableBase() = default;

  /// Cache the variable's batch shape
  virtual void cache(TorchShapeRef batch_shape);

  /// Setup the variable's views following another variable
  virtual void setup_views(const VariableBase * other);

  /// Setup the variable's views into blocks of the storage (AssemblyMode::INPLACE)
  virtual void setup_views(const LabeledVector * value,
                           const LabeledMatrix * deriv = nullptr,
                           const LabeledTensor3D * secderiv = nullptr);

  /// Setup the variable's input "view" (AssemblyMode::CONCATENATION)
  virtual void setup_views(BatchTensor * value);

  /// Setup the variable's output "views" (AssemblyMode::CONCATENATION)
  virtual void setup_views(const std::map<VariableName, size_t> & idx,
                           BatchTensor * value,
                           std::vector<BatchTensor> * deriv = nullptr,
                           std::vector<std::vector<BatchTensor>> * secderiv = nullptr);

  /// Set requires_grad for the underlying storage
  virtual void requires_grad_(bool req = true) = 0;

  /// Arguments
  const std::vector<VariableName> & args() const { return _args; }

  /// Add an argument
  void add_arg(const VariableBase & arg) { _args.push_back(arg.name()); }

  /// Clear arguments
  void clear_args() { _args.clear(); }

  /// Create a wrapper representing the derivative dy/dx
  Derivative d(const VariableBase & x);

  /// Create a wrapper representing the second derivative d2y/dx2
  Derivative d(const VariableBase & x1, const VariableBase & x2);

  ///@{ Accessors for storage
  const LabeledVector & value_storage() const;
  const LabeledMatrix & derivative_storage() const;
  const LabeledTensor3D & second_derivative_storage() const;
  /// @}

  /// Raw flattened variable value
  const BatchTensor & raw_value() const { return _raw_value; }

  /// Variable value of the logical shape
  virtual const BatchTensor tensor() const = 0;

  /// Name of this variable
  const VariableName & name() const { return _name; }

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
  const VariableName _name;

  /// Assembly mode
  // Note: This is a reference to the assembly mode of the model who declared this variable. If the
  // model changes assembly without taking care of variable reinitialization, all operations
  // regarding this variable become undefined -- segfault at best.
  const AssemblyMode & _assembly_mode;

  /// Batch shape of this variable
  TorchShape _batch_sizes;

  /// Names of the variables that this variable depends on
  std::vector<VariableName> _args;

  /// Members specific to AssemblyMode::INPLACE
  ///@{
  /// The raw (flattened) variable value
  BatchTensor _raw_value;
  /// The derivative of this variable w.r.t. arguments.
  std::map<VariableName, BatchTensor> _dvalue_d;
  /// The second derivative of this variable w.r.t. arguments.
  std::map<VariableName, std::map<VariableName, BatchTensor>> _d2value_d;
  /// The value storage that this variable is viewing into
  const LabeledVector * _value_storage;
  /// The derivative storage that this variable is viewing into
  const LabeledMatrix * _derivative_storage;
  /// The second derivative storage that this variable is viewing into
  const LabeledTensor3D * _second_derivative_storage;
  ///@}

  /// Members specific to AssemblyMode::CONCATENATION.
  /// To distinguish from their INPLACE counterparts, the members are prefixed
  /// with an additional underscore.
  ///@{
  /// Argument assembly indices
  const std::map<VariableName, size_t> * __args_idx;
  /// The raw (flattened) variable value
  BatchTensor * __raw_value;
  /// The derivative of this variable w.r.t. arguments.
  std::vector<BatchTensor> * __dvalue_d;
  /// The second derivative of this variable w.r.t. arguments.
  std::vector<std::vector<BatchTensor>> * __d2value_d;
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
  template <typename T2 = T, typename = typename std::enable_if_t<!std::is_same_v<BatchTensor, T2>>>
  Variable(const VariableName & name_in, const AssemblyMode & assembly_mode)
    : VariableBase(name_in, assembly_mode),
      _base_sizes(T::const_base_sizes)
  {
  }

  template <typename T2 = T, typename = typename std::enable_if_t<std::is_same_v<BatchTensor, T2>>>
  Variable(const VariableName & name_in,
           TorchShapeRef base_shape,
           const AssemblyMode & assembly_mode)
    : VariableBase(name_in, assembly_mode),
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

  virtual void requires_grad_(bool req = true) override
  {
    neml_assert_dbg(_assembly_mode == AssemblyMode::INPLACE,
                    "Cannot request AD in concatenation assembly mode");
    _value.requires_grad_(req);
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

  /// Set the variable value
  void operator=(const BatchTensor & val)
  {
    if (_assembly_mode == AssemblyMode::INPLACE)
      _value.index_put_({torch::indexing::Slice()},
                        val.batch_expand(batch_sizes()).base_reshape(base_sizes()));
    else if (_assembly_mode == AssemblyMode::CONCATENATION)
    {
      (*__raw_value) = val.batch_expand(batch_sizes()).base_reshape({base_storage()});
      _value = __raw_value->base_reshape(base_sizes());
    }
    else
      throw NEMLException("Unknown assembly mode");
  }

  /// Variable value of the logical shape
  const T & value() const { return _value; }

  /// Variable value of the logical shape
  virtual const BatchTensor tensor() const override { return value(); }

  /// Negation
  T operator-() const { return -value(); }

  operator T() const { return value(); }

  template <typename T2 = T, typename = typename std::enable_if_t<!std::is_same_v<T2, BatchTensor>>>
  operator BatchTensor() const
  {
    return tensor();
  }

  /// Set the batch shape and base shape according to \p val
  virtual void cache(TorchShapeRef batch_shape) override
  {
    VariableBase::cache(batch_shape);
    _sizes = utils::add_shapes(batch_shape, _base_sizes);
  }

protected:
  /// Base shape of this variable
  const TorchShape _base_sizes;

  /// Shape of this variable
  TorchShape _sizes;

  /// Variable value of the logical shape
  T _value;
};

class Derivative
{
public:
  Derivative(BatchTensor & val, TorchShapeRef batch_sizes, const AssemblyMode & assembly_mode)
    : _value(val),
      _batch_sizes(batch_sizes),
      _assembly_mode(assembly_mode)
  {
  }

  const BatchTensor & value() const { return _value; }

  void operator=(const BatchTensor & val);

private:
  BatchTensor & _value;

  TorchShapeRef _batch_sizes;

  const AssemblyMode & _assembly_mode;
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
