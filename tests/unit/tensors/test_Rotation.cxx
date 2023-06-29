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

#include "neml2/tensors/Rotation.h"
#include "neml2/tensors/Scalar.h"

using namespace neml2;

TEST_CASE("Rotation", "[Rotation]")
{
  SECTION("inverse rotations are in fact inverses")
  {
    SECTION("identity is zero")
    {
      Rotation a = Rotation::identity();
      REQUIRE(torch::allclose(a, torch::zeros_like(a)));
    }

    SECTION("unbatched")
    {
      Rotation a = Rotation::init(Scalar(1.2), Scalar(3.1), Scalar(-2.1));
      Rotation b = a.inverse();

      REQUIRE(torch::allclose(a * b, Rotation::identity()));
    }
    SECTION("batched")
    {
      Rotation a = Rotation::init(Scalar(torch::tensor({{1.2}, {-0.5}}, TorchDefaults)),
                                  Scalar(torch::tensor({{3.1}, {-1.6}}, TorchDefaults)),
                                  Scalar(torch::tensor({{-2.1}, {0.5}}, TorchDefaults)));
      Rotation b = a.inverse();

      REQUIRE(torch::allclose(a * b, Rotation::identity()));
    }
  }
  SECTION("test composition of rotations")
  {
    SECTION("unbatched")
    {
      Rotation a = Rotation::init(Scalar(1.2496889), Scalar(1.62862628), Scalar(7.59575411));
      Rotation b = Rotation::init(Scalar(-5.68010824), Scalar(-2.8011194), Scalar(15.25705169));
      Rotation c = Rotation::init(Scalar(-0.40390244), Scalar(0.61401441), Scalar(-0.27708492));

      REQUIRE(torch::allclose(a * b, c));
    }
    SECTION("batched")
    {
      Rotation a =
          Rotation::init(Scalar(torch::tensor({{1.2496889}, {-2.74440729}}, TorchDefaults)),
                         Scalar(torch::tensor({{1.62862628}, {-1.10086082}}, TorchDefaults)),
                         Scalar(torch::tensor({{7.59575411}, {-14.83201462}}, TorchDefaults)));
      Rotation b =
          Rotation::init(Scalar(torch::tensor({{-5.68010824}, {0.97525904}}, TorchDefaults)),
                         Scalar(torch::tensor({{-2.8011194}, {0.05227498}}, TorchDefaults)),
                         Scalar(torch::tensor({{15.25705169}, {-2.83462851}}, TorchDefaults)));
      Rotation c =
          Rotation::init(Scalar(torch::tensor({{-0.40390244}, {-0.05551478}}, TorchDefaults)),
                         Scalar(torch::tensor({{0.61401441}, {0.60802679}}, TorchDefaults)),
                         Scalar(torch::tensor({{-0.27708492}, {0.43687898}}, TorchDefaults)));

      REQUIRE(torch::allclose(a * b, c));
    }
  }

  SECTION("test conversion to matrix")
  {
    SECTION("unbatched")
    {
      Rotation a = Rotation::init(Scalar(1.2496889), Scalar(1.62862628), Scalar(7.59575411));
      auto Ap = R2(torch::tensor({{{-0.91855865, -0.1767767, 0.35355339},
                                   {0.30618622, -0.88388348, 0.35355339},
                                   {0.25, 0.4330127, 0.8660254}}},
                                 TorchDefaults));
      REQUIRE(torch::allclose(a.to_R2(), Ap));
    }
  }

  SECTION("rotate vectors")
  {
    SECTION("unbatched")
    {
      Rotation a = Rotation::init(Scalar(1.2496889), Scalar(1.62862628), Scalar(7.59575411));
      Vector v = Vector(torch::tensor({{1.0, -2.0, 3.0}}, TorchDefaults));

      Vector vp = Vector(torch::tensor({{0.495655, 3.13461, 1.98205}}, TorchDefaults));

      REQUIRE(torch::allclose(a.apply(v), vp));
    }
  }
}
