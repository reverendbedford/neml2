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
#include "neml2/tensors/tensors.h"

using namespace neml2;

TEST_CASE("Rot", "[tensors]")
{
  const auto & DTO = default_tensor_options();

  TorchShape B = {5, 3, 1, 2}; // batch shape

  SECTION("class Rot")
  {
    SECTION("identity")
    {
      auto a = Rot::identity(DTO);
      auto b = Vec(torch::rand(utils::add_shapes(B, 3), DTO));
      // Rotate by an identity rotation should be a no-op
      REQUIRE(torch::allclose(b.rotate(a), b));
    }

    SECTION("inverse")
    {
      auto a = Rot::fill(1.2, 3.1, -2.1, DTO);
      auto ab = a.batch_expand(B);
      auto b = Vec(torch::rand(utils::add_shapes(B, 3), DTO));
      REQUIRE(torch::allclose(b.rotate(a).rotate(a.inverse()), b));
      REQUIRE(torch::allclose(b.rotate(ab).rotate(ab.inverse()), b.batch_expand(B)));
    }

    SECTION("euler_rodrigues")
    {
      auto a = Rot::fill(0.13991834, 0.18234513, 0.85043991, DTO);
      auto A = R2::fill(-0.91855865,
                        -0.1767767,
                        0.35355339,
                        0.30618622,
                        -0.88388348,
                        0.35355339,
                        0.25,
                        0.4330127,
                        0.8660254,
                        DTO);
      REQUIRE(torch::allclose(a.euler_rodrigues(), A));
      REQUIRE(torch::allclose(a.batch_expand(B).euler_rodrigues(), A.batch_expand(B)));
    }

    SECTION("deuler_rodrigues")
    {
      auto a = Rot::fill(0.13991834, 0.18234513, 0.85043991, DTO);
      auto apply = [](const BatchTensor & x) { return Rot(x).euler_rodrigues(); };
      auto dA_da = finite_differencing_derivative(apply, a);

      REQUIRE(torch::allclose(a.deuler_rodrigues(), dA_da, 1.0e-4));
      REQUIRE(torch::allclose(a.batch_expand(B).deuler_rodrigues(), dA_da.batch_expand(B), 1.0e-4));
    }

    SECTION("shadow")
    {
      auto a = Rot::fill(1.2, 3.1, -2.1, DTO);
      auto ab = a.batch_expand(B);
      auto b = Rot::fill(-0.07761966, -0.20051746, 0.13583441, DTO);

      SECTION("defintion")
      {
        REQUIRE(torch::allclose(a.shadow(), b));
        REQUIRE(torch::allclose(ab.shadow(), b));
      }
      SECTION("concept") { REQUIRE(torch::allclose(a.euler_rodrigues(), b.euler_rodrigues())); }
      SECTION("derivative")
      {
        auto apply = [](const BatchTensor & x) { return Rot(x).shadow(); };
        auto dA = finite_differencing_derivative(apply, a);
        REQUIRE(torch::allclose(a.dshadow(), dA, 1.0e-4));
      }
    }

    SECTION("rotate")
    {
      auto r = Rot::fill(0.13991834, 0.18234513, 0.85043991, DTO);
      auto v = Rot::fill(-0.32366123, -0.15961206, 0.86937009, DTO);
      auto vp = Rot::fill(1.48720771, -2.26086024, 1.02025338, DTO);

      auto rb = r.batch_expand(B);
      auto vb = v.batch_expand(B);
      auto vpb = vp.batch_expand(B);

      REQUIRE(torch::allclose(v.rotate(r), vp));
      REQUIRE(torch::allclose(vb.rotate(rb), vpb));
      REQUIRE(torch::allclose(v.rotate(rb), vpb));
      REQUIRE(torch::allclose(vb.rotate(r), vpb));
    }

    SECTION("drotate")
    {
      auto r = Rot::fill(0.13991834, 0.18234513, 0.85043991, DTO);
      auto v = Rot::fill(-0.32366123, -0.15961206, 0.86937009, DTO);

      auto rb = r.batch_expand(B);
      auto vb = v.batch_expand(B);

      auto apply = [v](const BatchTensor & x) { return v.rotate(Rot(x)); };
      auto dvp_dr = finite_differencing_derivative(apply, r);
      auto dvp_drb = dvp_dr.batch_expand(B);

      REQUIRE(torch::allclose(v.drotate(r), dvp_dr, 1e-4));
      REQUIRE(torch::allclose(vb.drotate(rb), dvp_drb, 1e-4));
      REQUIRE(torch::allclose(v.drotate(rb), dvp_drb, 1e-4));
      REQUIRE(torch::allclose(vb.drotate(r), dvp_drb, 1e-4));
    }

    SECTION("drotate_self")
    {
      auto r = Rot::fill(0.13991834, 0.18234513, 0.85043991, DTO);
      auto v = Rot::fill(-0.32366123, -0.15961206, 0.86937009, DTO);

      auto apply = [r](const Rot & x) { return x.rotate(r); };
      auto dvp_dr = finite_differencing_derivative(apply, v);

      REQUIRE(torch::allclose(v.drotate_self(r), dvp_dr, 1e-4));
    }
  }

  SECTION("operator*")
  {
    auto a = Rot::fill(0.13991834, 0.18234513, 0.85043991, DTO);
    auto b = Rot::fill(-0.32366123, -0.15961206, 0.86937009, DTO);
    auto c = Rot::fill(1.48720771, -2.26086024, 1.02025338, DTO);
    REQUIRE(torch::allclose(a * b, c));
    REQUIRE(torch::allclose(a.batch_expand(B) * b, c.batch_expand(B)));
    REQUIRE(torch::allclose(a * b.batch_expand(B), c.batch_expand(B)));
    REQUIRE(torch::allclose(a.batch_expand(B) * b.batch_expand(B), c.batch_expand(B)));
  }
}
