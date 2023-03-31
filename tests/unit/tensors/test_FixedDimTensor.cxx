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

#include "neml2/tensors/FixedDimTensor.h"

using namespace neml2;

TEST_CASE("FixedDimTensors have the right shapes, construct blank", "[FixedDimTensors]")
{
  // 2 batch dimensions with sizes (10,2), base dimension (3,4)
  FixedDimTensor<2, 3, 4> A;
  A = A.batch_expand({10, 2});

  SECTION(" batch sizes are correct")
  {
    REQUIRE(A.batch_dim() == 2);
    REQUIRE(A.batch_sizes() == TorchShape({10, 2}));
  }

  SECTION(" base sizes are correct")
  {
    REQUIRE(A.base_dim() == 2);
    REQUIRE(A.base_sizes() == TorchShape({3, 4}));
  }
}

TEST_CASE("FixedDimTensors have the right shapes, construct with tensor", "[FixedDimTensors]")
{
  // 2 batch dimensions with sizes (10,2), base dimension (3,4)
  FixedDimTensor<2, 3, 4> A(torch::zeros({10, 2, 3, 4}));

  SECTION(" batch sizes are correct")
  {
    REQUIRE(A.batch_dim() == 2);
    REQUIRE(A.batch_sizes() == TorchShape({10, 2}));
  }

  SECTION(" base sizes are correct")
  {
    REQUIRE(A.base_dim() == 2);
    REQUIRE(A.base_sizes() == TorchShape({3, 4}));
  }
}

TEST_CASE("Not enough required dimensions", "[FixedDimTensor]")
{
#ifndef NDEBUG
  // Can't make this guy, as it won't have enough dimensions for the logical dimensions
  REQUIRE_THROWS(FixedDimTensor<2, 3, 4>(torch::zeros({10, 3, 4})));
#endif
}

TEST_CASE("FixedDimTensors can't be created with the wrong base dimensions", "[FixedDimTensor]")
{
#ifndef NDEBUG
  // Batch is okay, base dimension (5, 4) isn't what we expected (3, 4)
  REQUIRE_THROWS(FixedDimTensor<2, 3, 4>(torch::zeros({10, 2, 5, 4})));
#endif
}
