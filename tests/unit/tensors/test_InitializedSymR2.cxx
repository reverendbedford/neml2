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

TEST_CASE("AUTO", "[InitializedSymR2]")
{
  load_model("unit/tensors/test_InitializedSymR2.i");

  const auto & auto_0 = Factory::get_object<SymR2>("Tensors", "auto_0");
  const auto auto_0_correct = SymR2::zero().batch_expand(5);
  REQUIRE(torch::allclose(auto_0, auto_0_correct));

  const auto & auto_1 = Factory::get_object<SymR2>("Tensors", "auto_1");
  const auto auto_1_correct = SymR2::init(1).batch_expand(5);
  REQUIRE(torch::allclose(auto_1, auto_1_correct));

  const auto & auto_3 = Factory::get_object<SymR2>("Tensors", "auto_3");
  const auto auto_3_correct = SymR2::init(1, 2, 3).batch_expand(5);
  REQUIRE(torch::allclose(auto_3, auto_3_correct));

  const auto & auto_6 = Factory::get_object<SymR2>("Tensors", "auto_6");
  const auto auto_6_correct = SymR2::init(1, 2, 3, 4, 5, 6).batch_expand(5);
  REQUIRE(torch::allclose(auto_6, auto_6_correct));
}

TEST_CASE("ZERO", "[InitializedSymR2]")
{
  load_model("unit/tensors/test_InitializedSymR2.i");

  const auto & zero = Factory::get_object<SymR2>("Tensors", "zero");
  const auto zero_correct = SymR2::zero().batch_expand(5);
  REQUIRE(torch::allclose(zero, zero_correct));
}

TEST_CASE("IDENTITY", "[InitializedSymR2]")
{
  load_model("unit/tensors/test_InitializedSymR2.i");

  const auto & identity = Factory::get_object<SymR2>("Tensors", "identity");
  const auto identity_correct = SymR2::identity().batch_expand(5);
  REQUIRE(torch::allclose(identity, identity_correct));
}
