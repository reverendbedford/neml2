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

#include <catch2/catch_test_macros.hpp>

#include "utils.h"
#include "neml2/tensors/user_tensors/EmptyFixedDimTensor.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;

#define test_EmptyFixedDimTensor(tensor_type, tensor_name, batch_shape)                            \
  SECTION("Empty" #tensor_type)                                                                    \
  {                                                                                                \
    const auto tensor_name = Factory::get_object_ptr<tensor_type>("Tensors", #tensor_name);        \
    REQUIRE(tensor_name->batch_sizes() == batch_shape);                                            \
    REQUIRE(tensor_name->base_sizes() == tensor_type::const_base_sizes);                           \
  }                                                                                                \
  static_assert(true)

TEST_CASE("EmptyFixedDimTensor", "[tensors/user_tensors]")
{
  load_model("unit/tensors/user_tensors/test_EmptyFixedDimTensor.i");

  TensorShape B{2, 1};

  test_EmptyFixedDimTensor(Scalar, a, B);
  test_EmptyFixedDimTensor(Vec, b, B);
  test_EmptyFixedDimTensor(Rot, c, B);
  test_EmptyFixedDimTensor(R2, d, B);
  test_EmptyFixedDimTensor(SR2, e, B);
  test_EmptyFixedDimTensor(R3, f, B);
  test_EmptyFixedDimTensor(SFR3, g, B);
  test_EmptyFixedDimTensor(R4, h, B);
  test_EmptyFixedDimTensor(SSR4, i, B);
  test_EmptyFixedDimTensor(R5, j, B);
  test_EmptyFixedDimTensor(SSFR5, k, B);
}
