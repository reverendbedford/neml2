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

#include "neml2/models/VariableStore.h"
#include "neml2/models/Model.h"

namespace neml2
{
VariableStore::VariableStore(const OptionSet & options, Model * object)
  : _object(object),
    _options(options),
    _input_axis(declare_axis("input")),
    _output_axis(declare_axis("output"))
{
}

LabeledAxis &
VariableStore::declare_axis(const std::string & name)
{
  neml_assert(!_axes.has_key(name),
              "Trying to declare an axis named ",
              name,
              ", but an axis with the same name already exists.");

  auto axis = std::make_unique<LabeledAxis>();
  return *_axes.set_pointer(name, std::move(axis));
}

void
VariableStore::setup_layout()
{
  input_axis().setup_layout();
  output_axis().setup_layout();
}

VariableBase *
VariableStore::variable(const VariableName & name)
{
  return _variables.query_value(name);
}

const VariableBase *
VariableStore::variable(const VariableName & name) const
{
  return _variables.query_value(name);
}

void
VariableStore::setup_input_views(VariableStore * host)
{
  neml_assert_dbg(host || _object->host() == host,
                  "setup_input_views called on a non-host model without specifying the host as an "
                  "argument");
  for (const auto & varname : input_axis().variable_names())
  {
    auto var = variable(varname);
    // If I'm the host, reserve the variable here
    if (_object->host() == host)
      reserve(var);
    // otherwise, let the host reserve the variable
    else
      host->reserve(var);
  }
}

void
VariableStore::setup_output_views()
{
  for (const auto & varname : output_axis().variable_names())
  {
    auto var = variable(varname);
    reserve(var);
  }
}
} // namespace neml2
