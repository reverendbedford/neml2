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
class Derivative;
class SecondDerivative;

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

  /// Get the referencing variable
  virtual const VariableBase * ref() const = 0;

  /// Get the variable value
  virtual Tensor get() const = 0;

  /// Set the variable value (handles reshaping)
  virtual void set(const Tensor & val) = 0;

  /// Wrapper for assigning partial derivative
  Derivative d(const VariableBase & var);

  /// Wrapper for assigning second partial derivative
  SecondDerivative d(const VariableBase & var1, const VariableBase & var2);

  /// Initialize derivatives
  virtual void initialize_derivatives(const std::vector<const VariableBase *> & args,
                                      const torch::TensorOptions & options) = 0;

  /// Total derivatives
  virtual const std::map<VariableName, Tensor> & derivatives() const = 0;

  /// Total second derivatives
  virtual const std::map<VariableName, std::map<VariableName, Tensor>> &
  second_derivatives() const = 0;

  /// Apply first-order chain rule
  virtual void chain1(const Tensor & dy_du, const std::map<VariableName, Tensor> & du_dx) = 0;

  /// Apply second-order chain rule
  virtual void chain2a(const Tensor & d2y_du1u2,
                       const std::map<VariableName, Tensor> & du1_dx,
                       const std::map<VariableName, Tensor> & du2_dx) = 0;
  virtual void chain2b(const Tensor & dy_du,
                       const std::map<VariableName, std::map<VariableName, Tensor>> & d2u_dx2) = 0;

protected:
  /// Name of the variable
  const VariableName _name;

  /// The model which declared this variable
  const Model * const _owner;

  /// Variable ftype (role in a function definition)
  const FType _ftype;

  /// Variable tensor type
  const TensorType _type;
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
      _ref(nullptr)
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
      _ref(nullptr)
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
    _ref = var_ptr;
  }

  const VariableBase * ref() const override { return _ref; }

  Tensor get() const override { return _ref ? _ref->get() : Tensor(_value); }

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

  void initialize_derivatives(const std::vector<const VariableBase *> & args,
                              const torch::TensorOptions & options) override
  {
    for (auto x1 : args)
    {
      if (x1->name() == name())
        _derivs[x1->name()] = Tensor::identity(base_storage(), options);
      else
        _derivs[x1->name()] = Tensor::zeros({base_storage(), x1->base_storage()}, options);
      for (auto x2 : args)
        _sec_derivs[x1->name()][x2->name()] =
            Tensor::zeros({base_storage(), x1->base_storage(), x2->base_storage()}, options);
    }
  }

  const std::map<VariableName, Tensor> & derivatives() const override
  {
    if (_ref)
      return _ref->derivatives();
    return _derivs;
  }

  const std::map<VariableName, std::map<VariableName, Tensor>> & second_derivatives() const override
  {
    if (_ref)
      return _ref->second_derivatives();
    return _sec_derivs;
  }

  /// Suppressed constructor to prevent accidental dereferencing
  [[deprecated("Variable<T> must be assigned to references -- missing &")]] Variable(
      const Variable<T> &&)
  {
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
    neml_assert_dbg(!_ref, "Cannot assign value to a referencing variable.");
    _value = val;
  }

  /// Variable value
  const T & value() const { return _ref ? _ref->value() : _value; }

  /// Negation
  T operator-() const { return -value(); }

  operator T() const { return value(); }

  template <typename T2 = T, typename = typename std::enable_if_t<!std::is_same_v<T2, Tensor>>>
  operator Tensor() const
  {
    return value();
  }

  void chain1(const Tensor & dy_du, const std::map<VariableName, Tensor> & du_dx) override
  {
    for (const auto & [xvar, du_dx] : du_dx)
    {
      neml_assert_dbg(_derivs.count(xvar),
                      "Derivative of variable ",
                      name(),
                      " with respect to ",
                      xvar,
                      " is not initialized.");
      _derivs[xvar] = _derivs[xvar] + math::bmm(dy_du, du_dx);
    }
  }

  void chain2a(const Tensor & d2y_du1u2,
               const std::map<VariableName, Tensor> & du1,
               const std::map<VariableName, Tensor> & du2) override
  {
    for (const auto & [x1var, du1_dxj] : du1)
    {
      neml_assert_dbg(_sec_derivs.count(x1var),
                      "Second derivative of variable ",
                      name(),
                      " with respect to ",
                      x1var,
                      " is not initialized.");
      for (const auto & [x2var, du2_dxk] : du2)
      {
        neml_assert_dbg(_sec_derivs[x1var].count(x2var),
                        "Second derivative of variable ",
                        name(),
                        " with respect to ",
                        x1var,
                        " and ",
                        x2var,
                        " is not initialized.");
        _sec_derivs[x1var][x2var] =
            _sec_derivs[x1var][x2var] +
            Tensor(torch::einsum("...ipq,...pj,...qk", {d2y_du1u2, du1_dxj, du2_dxk}),
                   broadcast_batch_dim(d2y_du1u2, du1_dxj, du2_dxk));
      }
    }
  }

  void chain2b(const Tensor & dy_du,
               const std::map<VariableName, std::map<VariableName, Tensor>> & d2u) override
  {
    for (const auto & [x1var, d2u_dx1] : d2u)
    {
      neml_assert_dbg(_sec_derivs.count(x1var),
                      "Second derivative of variable ",
                      name(),
                      " with respect to ",
                      x1var,
                      " is not initialized.");
      for (const auto & [x2var, d2u_dx1x2] : d2u_dx1)
      {
        neml_assert_dbg(_sec_derivs[x1var].count(x2var),
                        "Second derivative of variable ",
                        name(),
                        " with respect to ",
                        x1var,
                        " and ",
                        x2var,
                        " is not initialized.");
        _sec_derivs[x1var][x2var] =
            _sec_derivs[x1var][x2var] + Tensor(torch::einsum("...ip,...pjk", {dy_du, d2u_dx1x2}),
                                               broadcast_batch_dim(dy_du, d2u_dx1x2));
      }
    }
  }

protected:
  /// Base shape of the variable
  const TensorShape _base_sizes;

  /// The variable referenced by this (nullptr if this is a storing variable)
  const Variable<T> * _ref;

  /// Variable value (undefined if this is a referencing variable)
  T _value;

  /// Derivatives of this variable with respect to other variables
  std::map<VariableName, Tensor> _derivs;

  /// Second derivatives of this variable with respect to other variables
  std::map<VariableName, std::map<VariableName, Tensor>> _sec_derivs;
};

class Derivative
{
public:
  Derivative()
    : _y(nullptr),
      _x(nullptr)
  {
  }

  Derivative(VariableBase & y, const VariableBase & x)
    : _y(&y),
      _x(&x)
  {
  }

  void operator=(const Tensor & val);

private:
  VariableBase * const _y;
  const VariableBase * const _x;
};

class SecondDerivative
{
public:
  SecondDerivative()
    : _y(nullptr),
      _x1(nullptr),
      _x2(nullptr)
  {
  }

  SecondDerivative(VariableBase & y, const VariableBase & x1, const VariableBase & x2)
    : _y(&y),
      _x1(&x1),
      _x2(&x2)
  {
  }

  void operator=(const Tensor & val);

private:
  VariableBase * const _y;
  const VariableBase * const _x1;
  const VariableBase * const _x2;
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
