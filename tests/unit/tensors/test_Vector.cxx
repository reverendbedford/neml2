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

#include "neml2/tensors/Vector.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/R2.h"

using namespace neml2;

TEST_CASE("Vector", "[Vector]")
{
  SECTION("Init from scalars")
  {
    Scalar s1 = Scalar(torch::tensor({{1.0}, {2.0}}, TorchDefaults));
    Scalar s2 = Scalar(torch::tensor({{3.0}, {4.0}}, TorchDefaults));
    Scalar s3 = Scalar(torch::tensor({{5.0}, {6.0}}, TorchDefaults));

    Vector v = Vector::init(s1, s2, s3);

    REQUIRE(torch::allclose(v, torch::tensor({{1.0, 3.0, 5.0}, {2.0, 4.0, 6.0}}, TorchDefaults)));
  }

  SECTION("math operations")
  {
    Vector v1 = Vector(torch::tensor({{0.04217105, 0.02973695, 0.69650092}}, TorchDefaults));
    Vector v2 = Vector(torch::tensor({{0.72223278, 0.06074174, 0.40130468}}, TorchDefaults));

    SECTION("Dot product")
    {
      Scalar res = Scalar(torch::tensor({{0.3117726719999968}}, TorchDefaults));

      SECTION("unbatched unbatched") { REQUIRE(torch::allclose(v1.dot(v2), res)); }

      SECTION("batched batched")
      {
        REQUIRE(torch::allclose(Vector(v1.batch_expand({10})).dot(Vector(v2.batch_expand({10}))),
                                res.batch_expand({10})));
      }

      SECTION("unbatched batched")
      {
        REQUIRE(torch::allclose(v1.dot(Vector(v2.batch_expand({10}))), res.batch_expand({10})));
      }

      SECTION("batched unbatched")
      {
        REQUIRE(torch::allclose(Vector(v1.batch_expand({10})).dot(v2), res.batch_expand({10})));
      }
    }

    SECTION("Cross product")
    {
      Vector res = Vector(torch::tensor({{-0.0303731, 0.48611236, -0.01891545}}, TorchDefaults));

      SECTION("unbatched unbatched") { REQUIRE(torch::allclose(v1.cross(v2), res)); }

      SECTION("batched batched")
      {
        REQUIRE(torch::allclose(Vector(v1.batch_expand({10})).cross(Vector(v2.batch_expand({10}))),
                                res.batch_expand({10})));
      }

      SECTION("unbatched batched")
      {
        REQUIRE(torch::allclose(v1.cross(Vector(v2.batch_expand({10}))), res.batch_expand({10})));
      }

      SECTION("batched unbatched")
      {
        REQUIRE(torch::allclose(Vector(v1.batch_expand({10})).cross(v2), res.batch_expand({10})));
      }
    }

    SECTION("Outer product")
    {
      R2 res = R2(torch::tensor({{{0.03045732, 0.00256154, 0.01692344},
                                  {0.021477, 0.00180627, 0.01193358},
                                  {0.5030358, 0.04230668, 0.27950908}}},
                                TorchDefaults));

      SECTION("unbatched unbatched") { REQUIRE(torch::allclose(v1.outer(v2), res)); }

      SECTION("batched batched")
      {
        REQUIRE(torch::allclose(Vector(v1.batch_expand({10})).outer(Vector(v2.batch_expand({10}))),
                                res.batch_expand({10})));
      }

      SECTION("unbatched batched")
      {
        REQUIRE(torch::allclose(v1.outer(Vector(v2.batch_expand({10}))), res.batch_expand({10})));
      }

      SECTION("batched unbatched")
      {
        REQUIRE(torch::allclose(Vector(v1.batch_expand({10})).outer(v2), res.batch_expand({10})));
      }
    }
  }
}
