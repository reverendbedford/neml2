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

#include "neml2/tensors/Quaternion.h"

using namespace neml2;

TEST_CASE("Quaternion", "[Quaternion]")
{
  SECTION("construct from scalars")
  {
    SECTION("unbatched")
    {
      Scalar q0(2.3);
      Scalar q1(1.2);
      Scalar q2(-0.8);
      Scalar q3(0.1);

      Quaternion result = Quaternion::init(q0, q1, q2, q3);
      Quaternion correct(torch::tensor({{2.3, 1.2, -0.8, 0.1}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched")
    {
      Scalar q0(torch::tensor({{2.3}, {3.4}}, TorchDefaults));
      Scalar q1(torch::tensor({{1.2}, {-4.1}}, TorchDefaults));
      Scalar q2(torch::tensor({{-0.8}, {1.1}}, TorchDefaults));
      Scalar q3(torch::tensor({{0.1}, {-0.1}}, TorchDefaults));

      Quaternion result = Quaternion::init(q0, q1, q2, q3);
      Quaternion correct(
          torch::tensor({{2.3, 1.2, -0.8, 0.1}, {3.4, -4.1, 1.1, -0.1}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
  }
  SECTION("construct identity")
  {
    Quaternion id = Quaternion::identity();

    REQUIRE(torch::allclose(id, torch::tensor({{1, 0, 0, 0}}, TorchDefaults)));
  }
  SECTION("Additive inverse, conjugate, multiplicative inverse")
  {
    Quaternion a = Quaternion(torch::tensor({{1.2, -3.4, 5.1, 0.1}}, TorchDefaults));
    SECTION("Additive inverse")
    {
      Quaternion b = -a;

      REQUIRE(torch::allclose(torch::tensor({{-1.2, 3.4, -5.1, -0.1}}, TorchDefaults), b));
    }
    SECTION("Conjugation")
    {
      Quaternion c = a.conj();

      REQUIRE(torch::allclose(torch::tensor({{1.2, 3.4, -5.1, -0.1}}, TorchDefaults), c));
    }
    SECTION("Multiplicative inverse, also multiplication")
    {
      Quaternion d = a.inverse();

      REQUIRE(torch::allclose(Quaternion::identity(), d * a));
    }
  }
  SECTION("Norms and inner products")
  {
    Quaternion a =
        Quaternion(torch::tensor({{-0.1, 0.9, 0.4, 1.1}, {1.2, -3.4, 5.1, 0.1}}, TorchDefaults));

    SECTION("Squared norm is really squared norm")
    {
      REQUIRE(torch::allclose(a.norm(), torch::sqrt(a.norm_sq())));
    }

    SECTION("Norm is correct")
    {
      REQUIRE(torch::allclose(a.norm(), torch::norm(a, 2, -1).unsqueeze(-1)));
    }

    SECTION("Squared normv is really squared normv")
    {
      REQUIRE(torch::allclose(a.normv(), torch::sqrt(a.normv_sq())));
    }

    SECTION("normv is correct")
    {
      REQUIRE(torch::allclose(
          a.normv(),
          torch::norm(a.index({torch::indexing::Ellipsis, torch::indexing::Slice({1, 4})}), 2, -1)
              .unsqueeze(-1)));
    }
  }
  SECTION("Scalar operations")
  {
    Quaternion a =
        Quaternion(torch::tensor({{-0.1, 0.9, 0.4, 1.1}, {1.2, -3.4, 5.1, 0.1}}, TorchDefaults));
    Scalar b = Scalar(torch::tensor({{2.0}, {3.0}}, TorchDefaults));
    SECTION("Scalar product")
    {
      REQUIRE(torch::allclose(
          a * b, torch::tensor({{-0.2, 1.8, 0.8, 2.2}, {3.6, -10.2, 15.3, 0.3}}, TorchDefaults)));
      REQUIRE(torch::allclose(a * b, b * a));
    }
    SECTION("Scalar division") { REQUIRE(torch::allclose(a / b, a * 1.0 / b)); }
  }
  SECTION("Exp, log, etc")
  {
    Quaternion a = Quaternion::init(0.46260799, 0.45213228, 0.21160485, 0.73266202);

    SECTION("exp correct versus NEML 1")
    {
      Quaternion ea = Quaternion::init(0.63207914, 0.39518801, 0.18495406, 0.64038613);
      REQUIRE(torch::allclose(exp(a), ea));
    }
    SECTION("exp is inverse of log") { REQUIRE(torch::allclose(a, exp(log(a)))); }
  }
}
