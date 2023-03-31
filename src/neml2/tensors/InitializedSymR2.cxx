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

#include "neml2/tensors/InitializedSymR2.h"

namespace neml2
{
register_NEML2_object(InitializedSymR2);

ParameterSet
InitializedSymR2::expected_params()
{
  ParameterSet params = NEML2Object::expected_params();
  params.set<std::vector<CrossRef<Scalar>>>("values") = {};
  params.set<TorchSize>("nbatch") = -1;
  params.set<std::string>("method") = "AUTO";
  return params;
}

InitializedSymR2::InitializedSymR2(const ParameterSet & params)
  : NEML2Object(params),
    SymR2(init().batch_expand(params.get<TorchSize>("nbatch")))
{
}

SymR2
InitializedSymR2::init() const
{
  auto method = input_parameters().get<std::string>("method");
  auto values = input_parameters().get<std::vector<CrossRef<Scalar>>>("values");

  if (method == "AUTO")
  {
    if (values.size() == 0)
      return SymR2::zero();
    else if (values.size() == 1)
      return SymR2::init(values[0]);
    else if (values.size() == 3)
      return SymR2::init(values[0], values[1], values[2]);
    else if (values.size() == 6)
      return SymR2::init(values[0], values[1], values[2], values[3], values[4], values[5]);
    else
      neml_assert(false, "fill method AUTO only supports values of size 0, 1, 3, and 6.");
  }
  else if (method == "ZERO")
  {
    neml_assert(values.empty(), "fill method ZERO ignores any provided value.");
    return SymR2::zero();
  }
  else if (method == "IDENTITY")
  {
    neml_assert(values.empty(), "fill method IDENTITY ignores any provided value.");
    return SymR2::identity();
  }
  else
    neml_assert(false, "Unrecognized fill method ", method);

  return SymR2::zero();
}

} // namespace neml2
