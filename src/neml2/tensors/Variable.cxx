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

#include "neml2/tensors/Variable.h"
#include "neml2/models/Model.h"

namespace neml2
{
VariableBase::VariableBase(const VariableName & name_in,
                           const Model * owner,
                           FType ftype,
                           TensorType type)
  : _name(name_in),
    _owner(owner),
    _ftype(ftype),
    _type(type)
{
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
  return Derivative(*this, var);
}

SecondDerivative
VariableBase::d(const VariableBase & var1, const VariableBase & var2)
{
  return SecondDerivative(*this, var1, var2);
}

void
VariableBase::initialize_derivatives(const std::set<VariableName> & args)
{
}

void
Derivative::operator=(const Tensor & val)
{
  neml_assert_dbg(_y, "Variable to take derivative of has not been initialized.");
  neml_assert_dbg(_x, "Variable to take derivative w.r.t. has not been initialized.");

  // Flatten partial derivative
  const auto dy_dx = val.base_reshape({_y->base_storage(), _x->base_storage()});

  // Apply chain rule
  for (const auto & [var, dx_dx0] : _x->derivatives())
  {
    const auto tmp = math::bmm(dy_dx, dx_dx0);
    auto & dy_dx0 = _y->derivatives()[var];
    if (dy_dx0.defined())
      dy_dx0 += tmp;
    else
      dy_dx0 = tmp;
  }

  // This may look weird:
  // The second order chain rule is
  //   (d2y/dx2)_{ijk} = (d2y/du2)_{ipq} (du/dx)_{pj} (du/dx)_{qk} + (dy/du)_{ip} (d2u/dx2)_{pjk}
  // Note the second term comes from the first partial derivative -- we need to accumulate it here
  // as we don't keep the first partial derivative
  for (const auto & [var1, d2x_dx0] : _x->second_derivatives())
    for (const auto & [var2, d2x_dx02] : d2x_dx0)
    {
      const auto tmp = Tensor(torch::einsum("...ip,...pjk", {dy_dx, d2x_dx02}),
                              broadcast_batch_dim(dy_dx, d2x_dx02));
      auto & d2y_dx02 = _y->second_derivatives()[var1][var2];
      if (d2y_dx02.defined())
        d2y_dx02 += d2y_dx02;
      else
        d2y_dx02 = d2y_dx02;
    }
}

void
SecondDerivative::operator=(const Tensor & val)
{
  neml_assert_dbg(_y, "Variable to take derivative of has not been initialized.");
  neml_assert_dbg(_x1, "Variable to take derivative w.r.t. has not been initialized.");
  neml_assert_dbg(_x2, "Variable to take derivative w.r.t. has not been initialized.");

  // Flatten second partial derivative
  const auto d2y_dx1x2 =
      val.base_reshape({_y->base_storage(), _x1->base_storage(), _x2->base_storage()});

  // Apply chain rule
  //   (d2y/dx2)_{ijk} = (d2y/du2)_{ipq} (du/dx)_{pj} (du/dx)_{qk} + (dy/du)_{ip} (d2u/dx2)_{pjk}
  // The second term is taken care of by Derivative::operator=
  for (const auto & [var1, dx1_dx0] : _x1->derivatives())
    for (const auto & [var2, dx2_dx0] : _x2->derivatives())
    {
      const auto tmp = Tensor(torch::einsum("...ipq,...pj,...qk", {d2y_dx1x2, dx1_dx0, dx2_dx0}),
                              broadcast_batch_dim(d2y_dx1x2, dx1_dx0, dx2_dx0));
      auto & d2y_dx02 = _y->second_derivatives()[var1][var2];
      if (d2y_dx02.defined())
        d2y_dx02 += tmp;
      else
        d2y_dx02 = tmp;
    }
}
}
