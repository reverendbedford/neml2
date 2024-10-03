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

#include <catch2/catch_test_macros.hpp>

#include "neml2/jit/StaticGraphFunction.h"
#include "neml2/tensors/Tensor.h"

TEST_CASE("jit", "[jit]")
{
  SECTION("tuple_to_stack")
  {
    auto tuple =
        std::make_tuple(torch::ones({1, 3, 2}), torch::ones({1, 2, 3}), torch::ones({5, 6}));

    auto stack = neml2::jit::tuple_to_stack(tuple);

    REQUIRE(torch::allclose(stack[0].toTensor(), torch::ones({1, 3, 2})));
    REQUIRE(torch::allclose(stack[1].toTensor(), torch::ones({1, 2, 3})));
    REQUIRE(torch::allclose(stack[2].toTensor(), torch::ones({5, 6})));
  }

  SECTION("stack_to_tuple")
  {
    torch::jit::Stack stack;
    stack.push_back(torch::ones({1, 3, 2}));
    stack.push_back(torch::ones({1, 2, 3}));
    stack.push_back(torch::ones({5, 6}));

    auto [a, b, c] = neml2::jit::stack_to_tuple<torch::Tensor, torch::Tensor, torch::Tensor>(stack);

    REQUIRE(torch::allclose(a, torch::ones({1, 3, 2})));
    REQUIRE(torch::allclose(b, torch::ones({1, 2, 3})));
    REQUIRE(torch::allclose(c, torch::ones({5, 6})));
  }
}

TEST_CASE("StaticGraphFunction", "[jit]")
{
  SECTION("basic")
  {
    auto f = [](torch::Tensor & x) -> std::tuple<torch::Tensor> { return {x + 1}; };
    auto f_jit = neml2::jit::StaticGraphFunction<std::tuple<torch::Tensor>, torch::Tensor>(
        "f", f, std::make_tuple(torch::ones({1, 2, 3})));
    auto [y] = f_jit(torch::full({2, 3}, 5.0));
    REQUIRE(torch::allclose(y, torch::full({2, 3}, 6.0)));
  }
}
