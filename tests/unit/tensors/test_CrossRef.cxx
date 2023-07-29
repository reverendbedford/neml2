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

#include <catch2/catch.hpp>

#include "utils.h"
#include "neml2/tensors/InitializedSymR2.h"

using namespace neml2;

TEST_CASE("cross-referencing Scalar", "[CrossRef]")
{
  load_model("unit/tensors/test_CrossRef.i");

  const auto & auto_3 = Factory::get_object<SymR2>("Tensors", "auto_3_crossref");

  const auto scalar1 = Scalar(torch::tensor({{1}, {2}, {3}, {4}, {5}}, default_tensor_options));
  const auto scalar2 = Scalar(torch::tensor({{5}, {6}, {7}, {8}, {9}}, default_tensor_options));
  const auto scalar3 =
      Scalar(torch::tensor({{-1}, {-2}, {-3}, {-4}, {-5}}, default_tensor_options));
  const auto auto_3_correct = SymR2::init(scalar1, scalar2, scalar3);

  REQUIRE(torch::allclose(auto_3, auto_3_correct));
}