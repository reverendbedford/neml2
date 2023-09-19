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

#include "neml2/tensors/Scalar.h"

using namespace neml2;

TEST_CASE("Unbatched Scalar", "[Scalar]")
{
  auto a = Scalar(2.5, default_tensor_options);

  SECTION("construct from plain data type")
  {
    torch::Tensor correct(torch::tensor({{2.5}}, default_tensor_options));
    REQUIRE(torch::allclose(a, correct));
  }

  SECTION("+ unbatched Scalar")
  {
    auto b = Scalar(3.1, default_tensor_options);
    auto result = a + b;
    auto correct = Scalar(5.6, default_tensor_options);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("+ batched Scalar")
  {
    TorchSize nbatch = 5;
    auto b = Scalar(3.1, default_tensor_options).batch_expand(nbatch);
    auto result = a + b;
    auto correct = Scalar(5.6, default_tensor_options).batch_expand(nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("- unbatched Scalar")
  {
    auto b = Scalar(3.1, default_tensor_options);
    auto result = a - b;
    auto correct = Scalar(-0.6, default_tensor_options);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("- batched Scalar")
  {
    int nbatch = 5;
    auto b = Scalar(3.1, default_tensor_options).batch_expand(nbatch);
    auto result = a - b;
    auto correct = Scalar(-0.6, default_tensor_options).batch_expand(nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("* unbatched Scalar")
  {
    auto b = Scalar(3.1, default_tensor_options);
    auto result = a * b;
    auto correct = Scalar(2.5 * 3.1, default_tensor_options);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("* batched Scalar")
  {
    int nbatch = 5;
    auto b = Scalar(3.1, default_tensor_options).batch_expand(nbatch);
    auto result = a * b;
    auto correct = Scalar(2.5 * 3.1, default_tensor_options).batch_expand(nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("/ unbatched Scalar")
  {
    auto b = Scalar(3.1, default_tensor_options);
    auto result = a / b;
    auto correct = Scalar(2.5 / 3.1, default_tensor_options);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("/ batched Scalar")
  {
    int nbatch = 5;
    auto b = Scalar(3.1, default_tensor_options).batch_expand(nbatch);
    auto result = a / b;
    auto correct = Scalar(2.5 / 3.1, default_tensor_options).batch_expand(nbatch);
    REQUIRE(torch::allclose(result, correct));
  }
}

TEST_CASE("Batched Scalar", "[Scalar]")
{
  int nbatch = 5;
  Scalar a = Scalar(2.5, default_tensor_options).batch_expand(nbatch);

  SECTION("+ unbatched Scalar")
  {
    auto b = Scalar(3.1, default_tensor_options);
    auto result = a + b;
    auto correct = Scalar(5.6, default_tensor_options).batch_expand(nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("+ batched Scalar")
  {
    Scalar b = Scalar(3.1, default_tensor_options).batch_expand(nbatch);
    auto result = a + b;
    auto correct = Scalar(5.6, default_tensor_options).batch_expand(nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("- unbatched Scalar")
  {
    auto b = Scalar(3.1, default_tensor_options);
    auto result = a - b;
    auto correct = Scalar(-0.6, default_tensor_options).batch_expand(nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("- batched Scalar")
  {
    Scalar b = Scalar(3.1, default_tensor_options).batch_expand(nbatch);
    auto result = a - b;
    auto correct = Scalar(-0.6, default_tensor_options).batch_expand(nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("* unbatched Scalar")
  {
    auto b = Scalar(3.1, default_tensor_options);
    auto result = a * b;
    auto correct = Scalar(2.5 * 3.1, default_tensor_options).batch_expand(nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("* batched Scalar")
  {
    Scalar b = Scalar(3.1, default_tensor_options).batch_expand(nbatch);
    auto result = a * b;
    auto correct = Scalar(2.5 * 3.1, default_tensor_options).batch_expand(nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("/ unbatched Scalar")
  {
    auto b = Scalar(3.1, default_tensor_options);
    auto result = a / b;
    auto correct = Scalar(2.5 / 3.1, default_tensor_options).batch_expand(nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("/ batched Scalar")
  {
    auto b = Scalar(3.1, default_tensor_options).batch_expand(nbatch);
    auto result = a / b;
    auto correct = Scalar(2.5 / 3.1, default_tensor_options).batch_expand(nbatch);
    REQUIRE(torch::allclose(result, correct));
  }
}

TEST_CASE("Scalar can't be created with semantically non-scalar tensors", "[Scalar]")
{
#ifndef NDEBUG
  // This can't happen as the tensor dimension is not (1,)
  REQUIRE_THROWS(Scalar(torch::zeros({2, 2})));
#endif
}
