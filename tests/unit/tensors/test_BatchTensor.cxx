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

#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include "neml2/tensors/BatchTensor.h"

using namespace neml2;

TEST_CASE("BatchTensors have the correct shapes", "[BatchTensors]")
{
  // Should have 2 batch dimensions and 3 base dimensions
  BatchTensor<2> A(torch::zeros({10, 5, 4, 11, 2}));

  SECTION(" batch sizes are correct")
  {
    REQUIRE(A.batch_dim() == 2);
    REQUIRE(A.batch_sizes() == TorchShape({10, 5}));
  }

  SECTION(" base sizes are correct")
  {
    REQUIRE(A.base_dim() == 3);
    REQUIRE(A.base_sizes() == TorchShape({4, 11, 2}));
  }
}

TEST_CASE("BatchTensors can't be created with few than the number of "
          "batch dimensions")
{
#ifndef NDEBUG
  REQUIRE_THROWS(BatchTensor<2>(torch::zeros({10})));
#endif
}
