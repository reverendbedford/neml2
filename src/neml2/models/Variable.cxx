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

#include "neml2/models/Variable.h"
#include "neml2/models/Model.h"
#include "neml2/tensors/tensors.h"

namespace neml2
{
VariableBase::VariableBase(const VariableName & name_in, Model * owner, TensorShapeRef list_shape)
  : _name(name_in),
    _owner(owner),
    _list_sizes(list_shape)
{
}

const Model &
VariableBase::owner() const
{
  neml_assert_dbg(_owner, "Owner of variable '", name(), "' has not been defined.");
  return *_owner;
}

Model &
VariableBase::owner()
{
  neml_assert_dbg(_owner, "Owner of variable '", name(), "' has not been defined.");
  return *_owner;
}

bool
VariableBase::is_state() const
{
  return _name.start_with("state");
}

bool
VariableBase::is_old_state() const
{
  return _name.start_with("old_state");
}

bool
VariableBase::is_force() const
{
  return _name.start_with("forces");
}

bool
VariableBase::is_old_force() const
{
  return _name.start_with("old_forces");
}

bool
VariableBase::is_residual() const
{
  return _name.start_with("residual");
}

bool
VariableBase::is_parameter() const
{
  return _name.start_with("parameters");
}

bool
VariableBase::is_solve_dependent() const
{
  return is_state() || is_residual() || is_parameter();
}

bool
VariableBase::is_dependent() const
{
  return !currently_solving_nonlinear_system() || is_solve_dependent();
}

Derivative
VariableBase::d(const VariableBase & var)
{
  neml_assert_dbg(owning(),
                  "Cannot assign derivative to a referencing variable '",
                  name(),
                  "' with respect to '",
                  var.name(),
                  "'.");
  return Derivative({assembly_storage(), var.assembly_storage()}, &_derivs[var.name()]);
}

Derivative
VariableBase::d(const VariableBase & var1, const VariableBase & var2)
{
  neml_assert_dbg(owning(),
                  "Cannot assign second derivative to a referencing variable '",
                  name(),
                  "' with respect to '",
                  var1.name(),
                  "' and '",
                  var2.name(),
                  "'.");
  return Derivative({assembly_storage(), var1.assembly_storage(), var2.assembly_storage()},
                    &_sec_derivs[var1.name()][var2.name()]);
}

void
VariableBase::request_AD(const VariableBase & u)
{
  owner().request_AD(*this, u);
}

void
VariableBase::request_AD(const std::vector<const VariableBase *> & us)
{
  for (const auto & u : us)
  {
    neml_assert(u, "Cannot request AD for a null variable.");
    owner().request_AD(*this, *u);
  }
}

void
VariableBase::request_AD(const VariableBase & u1, const VariableBase & u2)
{
  owner().request_AD(*this, u1, u2);
}

void
VariableBase::request_AD(const std::vector<const VariableBase *> & u1s,
                         const std::vector<const VariableBase *> & u2s)
{
  for (const auto & u1 : u1s)
    for (const auto & u2 : u2s)
    {
      neml_assert(u1, "Cannot request AD for a null variable.");
      neml_assert(u2, "Cannot request AD for a null variable.");
      owner().request_AD(*this, *u1, *u2);
    }
}

void
VariableBase::clear()
{
  neml_assert_dbg(owning(), "Cannot clear a referencing variable '", name(), "'.");
  _derivs.clear();
  _sec_derivs.clear();
}

void
VariableBase::apply_chain_rule(const DependencyResolver<Model, VariableName> & dep)
{
  for (const auto & [model, var] : dep.outbound_items())
    if (var == name())
    {
      _derivs = total_derivatives(dep, model, var);
      return;
    }
}

void
VariableBase::apply_second_order_chain_rule(const DependencyResolver<Model, VariableName> & dep)
{
  for (const auto & [model, var] : dep.outbound_items())
    if (var == name())
    {
      _sec_derivs = total_second_derivatives(dep, model, var);
      return;
    }
}

void
assign_or_add(Tensor & dest, const Tensor & val)
{
  if (dest.defined())
    dest = dest + val;
  else
    dest = val;
}

std::map<VariableName, Tensor>
VariableBase::total_derivatives(const DependencyResolver<Model, VariableName> & dep,
                                Model * model,
                                const VariableName & yvar) const
{
  std::map<VariableName, Tensor> derivs;

  for (const auto & [uvar, dy_du] : model->output_variable(yvar).derivatives())
  {
    if (dep.inbound_items().count({model, uvar}))
      assign_or_add(derivs[uvar], dy_du);
    else
      for (const auto & depu : dep.item_providers().at({model, uvar}))
        for (const auto & [xvar, du_dx] : total_derivatives(dep, depu.parent, uvar))
          assign_or_add(derivs[xvar], math::bmm(dy_du, du_dx));
  }

  return derivs;
}

std::map<VariableName, std::map<VariableName, Tensor>>
VariableBase::total_second_derivatives(const DependencyResolver<Model, VariableName> & dep,
                                       Model * model,
                                       const VariableName & yvar) const
{
  std::map<VariableName, std::map<VariableName, Tensor>> sec_derivs;

  for (const auto & [u1var, d2y_du1] : model->output_variable(yvar).second_derivatives())
    for (const auto & [u2var, d2y_du1u2] : d2y_du1)
    {
      if (dep.inbound_items().count({model, u1var}) && dep.inbound_items().count({model, u2var}))
        assign_or_add(sec_derivs[u1var][u2var], d2y_du1u2);
      else if (dep.inbound_items().count({model, u1var}))
        for (const auto & depu2 : dep.item_providers().at({model, u2var}))
          for (const auto & [x2var, du2_dxk] : total_derivatives(dep, depu2.parent, u2var))
            assign_or_add(sec_derivs[u1var][x2var],
                          Tensor(torch::einsum("...ijq,...qk", {d2y_du1u2, du2_dxk}),
                                 broadcast_batch_dim(d2y_du1u2, du2_dxk)));
      else if (dep.inbound_items().count({model, u2var}))
        for (const auto & depu1 : dep.item_providers().at({model, u1var}))
          for (const auto & [x1var, du1_dxj] : total_derivatives(dep, depu1.parent, u1var))
            assign_or_add(sec_derivs[x1var][u2var],
                          Tensor(torch::einsum("...ipk,...pj", {d2y_du1u2, du1_dxj}),
                                 broadcast_batch_dim(d2y_du1u2, du1_dxj)));
      else
        for (const auto & depu1 : dep.item_providers().at({model, u1var}))
          for (const auto & [x1var, du1_dxj] : total_derivatives(dep, depu1.parent, u1var))
            for (const auto & depu2 : dep.item_providers().at({model, u2var}))
              for (const auto & [x2var, du2_dxk] : total_derivatives(dep, depu2.parent, u2var))
                assign_or_add(
                    sec_derivs[x1var][x2var],
                    Tensor(torch::einsum("...ipq,...pj,...qk", {d2y_du1u2, du1_dxj, du2_dxk}),
                           broadcast_batch_dim(d2y_du1u2, du1_dxj, du2_dxk)));
    }

  for (const auto & [uvar, dy_du] : model->output_variable(yvar).derivatives())
    if (!dep.inbound_items().count({model, uvar}))
      for (const auto & depu : dep.item_providers().at({model, uvar}))
        for (const auto & [x1var, d2u_dx1] : total_second_derivatives(dep, depu.parent, uvar))
          for (const auto & [x2var, d2u_dx1x2] : d2u_dx1)
            assign_or_add(sec_derivs[x1var][x2var],
                          Tensor(torch::einsum("...ip,...pjk", {dy_du, d2u_dx1x2}),
                                 broadcast_batch_dim(dy_du, d2u_dx1x2)));

  return sec_derivs;
}

template <typename T>
TensorType
Variable<T>::type() const
{
  return TensorTypeEnum<T>::value;
}

template <typename T>
std::unique_ptr<VariableBase>
Variable<T>::clone(const VariableName & name, Model * owner) const
{
  if constexpr (std::is_same_v<T, Tensor>)
  {
    return std::make_unique<Variable<Tensor>>(
        name.empty() ? this->name() : name, owner ? owner : _owner, list_sizes(), base_sizes());
  }
  else
  {
    return std::make_unique<Variable<T>>(
        name.empty() ? this->name() : name, owner ? owner : _owner, list_sizes());
  }
}

template <typename T>
void
Variable<T>::ref(const VariableBase & var, bool ref_is_mutable)
{
  neml_assert(!_ref,
              "Variable '",
              name(),
              "' cannot reference another variable after it has been assigned a reference.");
  neml_assert(&var != this, "Variable '", name(), "' cannot reference itself.");
  neml_assert(var.ref() != this,
              "Variable '",
              name(),
              "' cannot reference a variable that is referencing itself.");
  const auto * var_ptr = dynamic_cast<const Variable<T> *>(var.ref());
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
  _ref_is_mutable = ref_is_mutable;
}

template <typename T>
void
Variable<T>::zero(const torch::TensorOptions & options)
{
  if (owning())
  {
    if constexpr (std::is_same_v<T, Tensor>)
      _value = T::zeros(base_sizes(), options);
    else
      _value = T::zeros(options);
  }
  else
  {
    neml_assert_dbg(_ref_is_mutable,
                    "Model '",
                    owner().name(),
                    "' is trying to zero a variable '",
                    name(),
                    "' declared by model '",
                    ref()->owner().name(),
                    "' , but the referenced variable is not mutable.");
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const_cast<VariableBase *>(ref())->zero(options);
  }
}

template <typename T>
void
Variable<T>::set(const Tensor & val)
{
  if (owning())
    _value = T(val.base_reshape(utils::add_shapes(list_sizes(), base_sizes())),
               utils::add_traceable_shapes(val.batch_sizes(), list_sizes()));
  else
  {
    neml_assert_dbg(_ref_is_mutable,
                    "Model '",
                    owner().name(),
                    "' is trying to assign value to a variable '",
                    name(),
                    "' declared by model '",
                    ref()->owner().name(),
                    "' , but the referenced variable is not mutable.");
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const_cast<VariableBase *>(ref())->set(val);
  }
}

template <typename T>
void
Variable<T>::set(const torch::Tensor & val, bool force)
{
  if (owning())
  {
    if constexpr (std::is_same_v<T, Tensor>)
      _value = T(val, val.dim() - base_dim());
    else
      _value = T(val);
  }
  else
  {
    neml_assert_dbg(_ref_is_mutable || force,
                    "Model '",
                    owner().name(),
                    "' is trying to assign value to a variable '",
                    name(),
                    "' declared by model '",
                    ref()->owner().name(),
                    "' , but the referenced variable is not mutable.");
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const_cast<VariableBase *>(ref())->set(val);
  }
}

template <typename T>
Tensor
Variable<T>::tensor() const
{
  if (owning())
  {
    neml_assert_dbg(_value.defined(), "Variable '", name(), "' has undefined value.");
    auto batch_sizes = _value.batch_sizes().slice(0, _value.batch_dim() - list_dim());
    return Tensor(_value, batch_sizes);
  }

  return ref()->tensor();
}

template <typename T>
void
Variable<T>::requires_grad_(bool req)
{
  if (owning())
    _value.requires_grad_(req);
  else
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const_cast<VariableBase *>(ref())->requires_grad_(req);
}

template <typename T>
void
Variable<T>::operator=(const Tensor & val)
{
  if (owning())
    _value = T(val);
  else
  {
    neml_assert_dbg(_ref_is_mutable,
                    "Model '",
                    owner().name(),
                    "' is trying to assign value to a variable '",
                    name(),
                    "' declared by model '",
                    ref()->owner().name(),
                    "' , but the referenced variable is not mutable.");
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    *const_cast<VariableBase *>(ref()) = val;
  }
}

template <typename T>
void
Variable<T>::clear()
{
  if (owning())
  {
    VariableBase::clear();
    _value = T();
  }
  else
  {
    neml_assert_dbg(_ref_is_mutable,
                    "Model '",
                    owner().name(),
                    "' is trying to clear a variable '",
                    name(),
                    "' declared by model '",
                    ref()->owner().name(),
                    "' , but the referenced variable is not mutable.");
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const_cast<VariableBase *>(ref())->clear();
  }
}

#define INSTANTIATE_VARIABLE(T) template class Variable<T>
FOR_ALL_TENSORBASE(INSTANTIATE_VARIABLE);

void
Derivative::operator=(const Tensor & val)
{
  *_deriv = val.base_reshape(_base_sizes);
}
}
