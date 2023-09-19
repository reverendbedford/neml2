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

#include "neml2/tensors/Vec.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/R2.h"
#include "neml2/tensors/SymR2.h"

using namespace neml2;

TEST_CASE("R2", "[R2]")
{
  SECTION("init zero")
  {
    REQUIRE(torch::allclose(R2::identity(), torch::eye(3, default_tensor_options).unsqueeze(0)));
  }

  SECTION("init identity")
  {
    REQUIRE(torch::allclose(R2::zero(), torch::zeros({3, 3}, default_tensor_options).unsqueeze(0)));
  }

  SECTION("init SymR2")
  {
    SymR2 s(
        torch::tensor({{1.0, 1.0, 1.0, sqrt(2.0), sqrt(2.0), sqrt(2.0)}}, default_tensor_options));
    REQUIRE(torch::allclose(R2::init_sym(s), torch::ones({3, 3}, default_tensor_options)));
  }

  SECTION("sym")
  {
    R2 u(torch::rand({1, 3, 3}, default_tensor_options));
    REQUIRE(torch::allclose(SymR2::init(u), u.sym()));
  }

  SECTION("math operations")
  {
    R2 A(torch::tensor({{{0.89072077, 0.82632195, 0.04234413},
                         {0.85614465, 0.9737414, 0.9491076},
                         {0.07230831, 0.49570559, 0.42566357}}},
                       default_tensor_options));

    SECTION("transpose")
    {
      R2 res(torch::tensor({{{0.89072077, 0.85614465, 0.07230831},
                             {0.82632195, 0.9737414, 0.49570559},
                             {0.04234413, 0.9491076, 0.42566357}}},
                           default_tensor_options));
      SECTION("unbatched") { REQUIRE(torch::allclose(A.transpose(), res)); }

      SECTION("batched")
      {
        REQUIRE(torch::allclose(R2(A.batch_expand_copy({10})).transpose(),
                                res.batch_expand_copy({10})));
      }
    }

    SECTION("R2-R2")
    {
      R2 B(torch::tensor({{{0.94487948, 0.15144116, 0.03378262},
                           {0.51449125, 0.56508379, 0.06392479},
                           {0.46572483, 0.00443049, 0.42061576}}},
                         default_tensor_options));
      R2 res(torch::tensor({{{1.28647989, 0.60202053, 0.10072394},
                             {1.75195791, 0.68410603, 0.49037864},
                             {0.52160092, 0.29295155, 0.21317145}}},
                           default_tensor_options));
      SECTION("unbatched unbatched") { REQUIRE(torch::allclose(A * B, res)); }

      SECTION("batched batched")
      {
        REQUIRE(torch::allclose(R2(A.batch_expand({10})) * R2(B.batch_expand({10})),
                                res.batch_expand({10})));
      }

      SECTION("unbatched batched")
      {
        REQUIRE(torch::allclose(A * R2(B.batch_expand({10})), res.batch_expand({10})));
      }

      SECTION("batched unbatched")
      {
        REQUIRE(torch::allclose(R2(A.batch_expand({10})) * B, res.batch_expand({10})));
      }
    }

    SECTION("R2-Vec")
    {
      Vec B(torch::tensor({{0.05425937, 0.55065082, 0.24673347}}, default_tensor_options));
      Vec res(torch::tensor({{0.51379251, 0.81682197, 0.38190954}}, default_tensor_options));

      SECTION("unbatched unbatched") { REQUIRE(torch::allclose(A * B, res)); }

      SECTION("batched batched")
      {
        REQUIRE(torch::allclose(R2(A.batch_expand({10})) * Vec(B.batch_expand({10})),
                                res.batch_expand({10})));
      }

      SECTION("unbatched batched")
      {
        REQUIRE(torch::allclose(A * Vec(B.batch_expand({10})), res.batch_expand({10})));
      }

      SECTION("batched unbatched")
      {
        REQUIRE(torch::allclose(R2(A.batch_expand({10})) * B, res.batch_expand({10})));
      }
    }

    SECTION("R2-scalar")
    {
      Scalar B(torch::tensor({{0.7309657192592477}}, default_tensor_options));
      R2 res(torch::tensor({{{0.65108635, 0.60401302, 0.03095211},
                             {0.62581239, 0.71177159, 0.69376512},
                             {0.0528549, 0.36234379, 0.31114547}}},
                           default_tensor_options));

      SECTION("unbatched unbatched") { REQUIRE(torch::allclose(A * B, res)); }

      SECTION("batched batched")
      {
        REQUIRE(torch::allclose(R2(A.batch_expand({10})) * Scalar(B.batch_expand({10})),
                                res.batch_expand({10})));
      }

      SECTION("unbatched batched")
      {
        REQUIRE(torch::allclose(A * Scalar(B.batch_expand({10})), res.batch_expand({10})));
      }

      SECTION("batched unbatched")
      {
        REQUIRE(torch::allclose(R2(A.batch_expand({10})) * B, res.batch_expand({10})));
      }
    }
  }
}
