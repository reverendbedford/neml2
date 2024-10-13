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
}
