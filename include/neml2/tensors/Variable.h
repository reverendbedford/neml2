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
#include "neml2/tensors/StorageTensor.h"
#include "neml2/tensors/LabeledTensor.h"

namespace neml2
{
using VariableName = LabeledAxisAccessor;

class VariableBase
{
public:
  VariableBase(const VariableName & name_in, const AssemblyMode & assembly_mode)
    : _name(name_in),
      _assembly_mode(assembly_mode)
  {
  }

  virtual ~VariableBase() = default;

  /// Methods for setting up the variable
  ///@{
  /// Cache the variable's batch shape
  virtual void cache(TorchShapeRef batch_shape);
  /// Setup the variable's views following another variable
  virtual void setup_views(VariableBase * other) = 0;
  /// Setup the variable's views into blocks of the storage (AssemblyMode::INPLACE)
  virtual void setup_views(StorageTensor<1> * value,
                           StorageTensor<2> * deriv = nullptr,
                           StorageTensor<3> * secderiv = nullptr) = 0;
  ///@}

  /// Variable metadata accessors
  ///@{
  /// Name of this variable
  const VariableName & name() const { return _name; }
  /// Total shape
  virtual TorchShapeRef sizes() const = 0;
  /// Batch shape
  TorchShapeRef batch_sizes() const { return _batch_sizes; }
  /// Base shape
  virtual TorchShapeRef base_sizes() const = 0;
  /// Batch dimension
  TorchSize batch_dim() const { return batch_sizes().size(); }
  /// Base dimension
  TorchSize base_dim() const { return base_sizes().size(); }
  /// Base storage
  TorchSize base_storage() const { return utils::storage_size(base_sizes()); }
  ///@}

  /// Variable value getters
  ///@{
  /// Raw flattened variable value
  virtual const BatchTensor & raw_value() const = 0;
  /// Variable value as a BatchTensor with the logical base shape
  BatchTensor tensor() const { return raw_value().base_reshape(base_sizes()); }
  ///@}

  /// Variable value modifiers
  ///@{
  /// Set requires_grad for the underlying storage
  virtual void requires_grad_(bool req = true) = 0;
  /// Assignment operator
  virtual void operator=(const BatchTensor & val) = 0;
  ///@}

  /// Derivative accessors
  ///@{
  /// Create a wrapper representing the derivative dy/dx
  StorageTensor<2>::View<BatchTensor> & d(const VariableBase & x);
  /// Create a wrapper representing the second derivative d2y/dx2
  StorageTensor<3>::View<BatchTensor> & d(const VariableBase & x1, const VariableBase & x2);
  /// Get all derivatives
  const std::map<VariableName, StorageTensor<2>::View<BatchTensor> *> & derivatives();
  /// Get all second derivatives
  const std::map<VariableName, std::map<VariableName, StorageTensor<3>::View<BatchTensor> *>> &
  second_derivatives();
  ///@}

protected:
  /// Name of the variable
  const VariableName _name;

  /// Assembly mode
  const AssemblyMode & _assembly_mode;

  /// Batch shape of this variable
  TorchShape _batch_sizes;

  /// The derivative of this variable w.r.t. arguments.
  std::map<VariableName, StorageTensor<2>::View<BatchTensor> *> _deriv_views;
  /// The second derivative of this variable w.r.t. arguments.
  std::map<VariableName, std::map<VariableName, StorageTensor<3>::View<BatchTensor> *>>
      _sec_deriv_views;
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

  /// Dummy operators to prevent accidental dereferencing
  ///@{
  /// Suppressed constructor
  [[deprecated("Variable<T> must be assigned to references -- missing &")]] Variable(
      const Variable<T> &)
  {
  }
  /// Suppressed assignment operator
  [[deprecated("Variable<T> must be assigned to references -- missing &")]] void
  operator=(const Variable<T> &)
  {
  }
  ///@}

  /// Methods for setting up the variable
  ///@{
  /// Cache the variable's batch shape
  virtual void cache(TorchShapeRef batch_shape) override;
  /// Setup the variable's views following another variable
  virtual void setup_views(VariableBase * other) override;
  /// Setup the variable's views into blocks of the storage (AssemblyMode::INPLACE)
  virtual void setup_views(StorageTensor<1> * value,
                           StorageTensor<2> * deriv = nullptr,
                           StorageTensor<3> * secderiv = nullptr) override;
  ///@}

  /// Variable metadata accessors
  ///@{
  /// Base shape
  virtual TorchShapeRef sizes() const override { return _sizes; }
  /// Total shape
  virtual TorchShapeRef base_sizes() const override { return _base_sizes; }
  ///@}

  /// Variable value getters
  ///@{
  /// Raw flattened variable value
  virtual const BatchTensor & raw_value() const override { return _view->raw_value(); }
  /// Variable value of the logical shape
  const T & value() const { return _view->value(); }
  StorageTensor<1>::View<T> * view() { return _view; }
  ///@}

  /// Variable value modifiers
  ///@{
  /// Set requires_grad for the underlying storage
  virtual void requires_grad_(bool req = true) override;
  /// Assignment operator
  virtual void operator=(const BatchTensor & val) override;
  ///@}

  /// Variable value getters and conversion operators
  ///@{
  /// Negation
  T operator-() const;
  /// Conversion operator
  operator T() const;
  /// Conversion operator
  template <typename T2 = T, typename = typename std::enable_if_t<!std::is_same_v<T2, BatchTensor>>>
  operator BatchTensor() const;
  ///@}

protected:
  /// Total shape
  TorchShape _sizes;

  /// Base shape
  const TorchShape _base_sizes;

  /// The variable value view
  StorageTensor<1>::View<T> * _view;
};

///////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////

template <typename T>
void
Variable<T>::cache(TorchShapeRef batch_shape)
{
  VariableBase::cache(batch_shape);
  _sizes = utils::add_shapes(batch_shape, _base_sizes);
}

template <typename T>
void
Variable<T>::setup_views(VariableBase * other)
{
  auto other_s = dynamic_cast<Variable<T> *>(other);
  neml_assert(other_s, "Failed to cast variable");

  // value
  _view = &other_s->view()->clone();

  // derivatives
  _deriv_views.clear();
  for (auto && [arg, view] : other->derivatives())
    _deriv_views[arg] = &view->clone();

  // second derivatives
  _sec_deriv_views.clear();
  for (auto && [arg1, arg2_views] : other->second_derivatives())
    for (auto && [arg2, view] : arg2_views)
      _sec_deriv_views[arg1][arg2] = &view->clone();
}

template <typename T>
void
Variable<T>::setup_views(StorageTensor<1> * value,
                         StorageTensor<2> * deriv,
                         StorageTensor<3> * secderiv)
{
  if (_assembly_mode == AssemblyMode::INPLACE)
  {
    if (value)
    {
      auto value_s = dynamic_cast<LabeledVector *>(value);
      neml_assert(value_s, "Failed to cast value storage");
      _view = &value_s->view<T>({name()});
    }
    if (deriv)
    {
      auto deriv_s = dynamic_cast<LabeledMatrix *>(value);
      neml_assert(deriv_s, "Failed to cast derivative storage");
      for (auto arg : deriv_s->axis(1).variable_accessors(true))
        _deriv_views[arg] = &deriv_s->view<BatchTensor>({name(), arg});
    }
    if (secderiv)
    {
      auto secderiv_s = dynamic_cast<LabeledTensor3D *>(secderiv);
      neml_assert(secderiv_s, "Failed to cast second derivative storage");
      for (auto arg1 : secderiv_s->axis(1).variable_accessors(true))
        for (auto arg2 : secderiv_s->axis(2).variable_accessors(true))
          _sec_deriv_views[arg1][arg2] = &secderiv_s->view<BatchTensor>({name(), arg1, arg2});
    }
  }
  else if (_assembly_mode == AssemblyMode::CONCATENATION)
  {
    throw NEMLException("Not implemented");
  }
  else
    throw NEMLException("Unknown assembly mode");
}

template <typename T>
void
Variable<T>::requires_grad_(bool req)
{
  _view->requires_grad_(req);
}

template <typename T>
void
Variable<T>::operator=(const BatchTensor & val)
{
  *_view = val;
}

template <typename T>
T
Variable<T>::operator-() const
{
  return -_view->value();
}

template <typename T>
Variable<T>::operator T() const
{
  return _view->value();
}

template <typename T>
template <typename T2, typename>
Variable<T>::operator BatchTensor() const
{
  return _view->value();
}

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
