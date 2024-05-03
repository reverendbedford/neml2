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
#include "neml2/tensors/user_tensors/Fill3DVec.h"

using namespace neml2;

TEST_CASE("Fill3DVec", "[tensors/user_tensors]")
{
  load_model("unit/tensors/user_tensors/test_Fill3DVec.i");

  const auto valid_1 = Factory::get_object_ptr<Vec>("Tensors", "v1");
  const auto correct_1 = Vec::fill(1.0, 2.0, 3.0);
  REQUIRE(torch::allclose(*valid_1, correct_1));

  const auto valid_4 = Factory::get_object_ptr<Vec>("Tensors", "v4");
  const auto correct_4 =
      Vec(torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}},
                        default_tensor_options()));
  REQUIRE(torch::allclose(*valid_4, correct_4));

  REQUIRE_THROWS(Factory::get_object_ptr<Vec>("Tensors", "invalid"));
}
