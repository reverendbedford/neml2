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

#include "VTestTimeSeries.h"
#include "VTestParser.h"

namespace neml2
{
register_NEML2_object(VTestTimeSeries);

OptionSet
VTestTimeSeries::expected_options()
{
  OptionSet options = NEML2Object::expected_options();
  options.set<std::string>("vtest");
  options.set<std::string>("variable");
  options.set<std::string>("variable_type");
  return options;
}

VTestTimeSeries::VTestTimeSeries(const OptionSet & options)
  : NEML2Object(options),
    torch::Tensor(init(options))
{
}

torch::Tensor
VTestTimeSeries::init(const OptionSet & options) const
{
  VTestParser table(options.get<std::string>("vtest"));
  auto var = options.get<std::string>("variable");
  auto var_type = options.get<std::string>("variable_type");

  if (var_type == "SCALAR")
    return table[var].unsqueeze(-1);
  else if (var_type == "SYMR2")
  {
    auto val_xx = table[var + "_xx"].unsqueeze(-1);
    auto val_yy = table[var + "_yy"].unsqueeze(-1);
    auto val_zz = table[var + "_zz"].unsqueeze(-1);
    auto val_yz = table[var + "_yz"].unsqueeze(-1);
    auto val_xz = table[var + "_xz"].unsqueeze(-1);
    auto val_xy = table[var + "_xy"].unsqueeze(-1);
    // The vtest format provides SR2 in Mandel notation already
    return torch::stack({val_xx, val_yy, val_zz, val_yz, val_xz, val_xy}, -1);
  }

  neml_assert("Unrecognized variable_type: ", var_type);
  return torch::Tensor();
}
} // namespace neml2
