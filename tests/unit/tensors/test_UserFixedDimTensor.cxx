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
#include "neml2/tensors/UserFixedDimTensor.h"

using namespace neml2;

TEST_CASE("UserScalar", "[UserFixedDimTensor]")
{
  load_model("unit/tensors/test_UserFixedDimTensor.i");
  const auto & a = Factory::get_object<Scalar>("Tensors", "a");
  const auto a_correct = torch::tensor({{1}, {2}, {3}, {4}, {5}}, default_tensor_options);
  REQUIRE(torch::allclose(a, a_correct));
}

TEST_CASE("UserSymR2", "[UserFixedDimTensor]")
{
  load_model("unit/tensors/test_UserFixedDimTensor.i");
  const auto & b = Factory::get_object<SymR2>("Tensors", "b");
  const auto b_correct =
      torch::tensor({{1, 2, 3, 4, 5, 6}, {-1, -2, -3, -4, -5, -6}}, default_tensor_options);
  REQUIRE(torch::allclose(b, b_correct));
}

TEST_CASE("UserSymSymR4", "[UserFixedDimTensor]")
{
  load_model("unit/tensors/test_UserFixedDimTensor.i");
  const auto & c = Factory::get_object<SymSymR4>("Tensors", "c");
  const auto c_correct = torch::tensor({{{1, 2, 3, 4, 5, 6},
                                         {7, 8, 9, 10, 11, 12},
                                         {13, 14, 15, 16, 17, 18},
                                         {19, 20, 21, 22, 23, 24},
                                         {25, 26, 27, 28, 29, 30},
                                         {31, 32, 33, 34, 35, 36}}},
                                       default_tensor_options);
  REQUIRE(torch::allclose(c, c_correct));
}
