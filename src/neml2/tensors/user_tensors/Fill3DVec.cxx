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

#include "neml2/tensors/user_tensors/Fill3DVec.h"

namespace neml2
{
register_NEML2_object(Fill3DVec);

OptionSet
Fill3DVec::expected_options()
{
  OptionSet options = NEML2Object::expected_options();
  options.set<std::vector<Real>>("values");
  return options;
}

Fill3DVec::Fill3DVec(const OptionSet & options)
  : Vec(fill(options.get<std::vector<Real>>("values"))),
    NEML2Object(options)
{
}

Vec
Fill3DVec::fill(const std::vector<Real> & values) const
{
  if ((values.size() % 3) != 0)
    neml_assert(false, "Number of provided values must be a multiple of three!");

  return Vec(torch::tensor(values, default_tensor_options()).reshape({-1, 3}));
}
} // namespace neml2
