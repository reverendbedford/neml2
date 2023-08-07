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

#include "neml2/tensors/SymSymR4.h"
#include "neml2/tensors/SymR2.h"

using namespace torch::indexing;

using namespace neml2;

TEST_CASE("SymSymR4", "[SymSymR4]")
{
  SECTION("initialize symmetric identity")
  {
    SymSymR4 result = SymSymR4::init_identity_sym();
    SymSymR4 correct(torch::tensor({{{1, 0, 0, 0, 0, 0},
                                     {0, 1, 0, 0, 0, 0},
                                     {0, 0, 1, 0, 0, 0},
                                     {0, 0, 0, 1, 0, 0},
                                     {0, 0, 0, 0, 1, 0},
                                     {0, 0, 0, 0, 0, 1}}},
                                   default_tensor_options));
    REQUIRE(torch::allclose(result, correct));
    // An additional sanity check: the symmetric identity tensor is a projector, hence
    REQUIRE(torch::allclose(result * result, result));
  }

  SECTION("initialize volumetric identity")
  {
    SymSymR4 result = SymSymR4::init_identity_vol();
    SymSymR4 correct(BatchTensor<1>(torch::tensor({{{1, 1, 1, 0, 0, 0},
                                                    {1, 1, 1, 0, 0, 0},
                                                    {1, 1, 1, 0, 0, 0},
                                                    {0, 0, 0, 0, 0, 0},
                                                    {0, 0, 0, 0, 0, 0},
                                                    {0, 0, 0, 0, 0, 0}}},
                                                  default_tensor_options)) /
                     3.0);
    REQUIRE(torch::allclose(result, correct));
    // An additional sanity check: the volumetric identity tensor is a projector, hence
    REQUIRE(torch::allclose(result * result, result));
    // Yet another sanity check: the volumetric identity tensor should project a second order tensor
    // onto its volumetric part, hence
    SymR2 A = SymR2::init(3, 2, 1, 5, 6, 7);
    REQUIRE(torch::allclose(result * A, A.vol()));
  }

  SECTION("initialize deviatoric identity")
  {
    SymSymR4 result = SymSymR4::init_identity_dev();
    SymSymR4 correct(torch::tensor({{{2. / 3., -1. / 3., -1. / 3., 0., 0., 0.},
                                     {-1. / 3., 2. / 3., -1. / 3., 0., 0., 0.},
                                     {-1. / 3., -1. / 3., 2. / 3., 0., 0., 0.},
                                     {0., 0., 0., 1., 0., 0.},
                                     {0., 0., 0., 0., 1., 0.},
                                     {0., 0., 0., 0., 0., 1.}}},
                                   default_tensor_options));
    REQUIRE(torch::allclose(result, correct));
    // An additional sanity check: the deviatoric identity tensor is a projector, hence
    REQUIRE(torch::allclose(result * result, result));
    // Yet another sanity check: the deviatoric identity tensor should project a second order tensor
    // onto its deviatoric part, hence
    SymR2 A = SymR2::init(3, 2, 1, 5, 6, 7);
    REQUIRE(torch::allclose(result * A, A.dev()));
  }

  SECTION("elasticity tensor from E and nu")
  {
    SECTION("unbatched")
    {
      Scalar E(100, default_tensor_options);
      Scalar nu(0.3, default_tensor_options);
      SymSymR4 result = SymSymR4::init_isotropic_E_nu(E, nu);
      SymSymR4 correct(torch::tensor({{{134.6154, 57.6923, 57.6923, 0.0000, 0.0000, 0.0000},
                                       {57.6923, 134.6154, 57.6923, 0.0000, 0.0000, 0.0000},
                                       {57.6923, 57.6923, 134.6154, 0.0000, 0.0000, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 76.9231, 0.0000, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 0.0000, 76.9231, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 76.9231}}},
                                     default_tensor_options));
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched")
    {
      Scalar E(torch::tensor({{100}, {200}}, default_tensor_options));
      Scalar nu(torch::tensor({{0.3}, {0.25}}, default_tensor_options));
      SymSymR4 result = SymSymR4::init_isotropic_E_nu(E, nu);
      SymSymR4 correct(torch::tensor({{{134.6154, 57.6923, 57.6923, 0.0000, 0.0000, 0.0000},
                                       {57.6923, 134.6154, 57.6923, 0.0000, 0.0000, 0.0000},
                                       {57.6923, 57.6923, 134.6154, 0.0000, 0.0000, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 76.9231, 0.0000, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 0.0000, 76.9231, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 76.9231}},
                                      {{240., 80., 80., 0.0000, 0.0000, 0.0000},
                                       {80., 240., 80., 0.0000, 0.0000, 0.0000},
                                       {80., 80., 240., 0.0000, 0.0000, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 160., 0.0000, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 0.0000, 160., 0.0000},
                                       {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 160.}}},
                                     default_tensor_options));
      REQUIRE(torch::allclose(result, correct));
    }
  }

  SECTION("ijkl,kl->ij")
  {
    SymSymR4 C_unbatched = SymSymR4::init_isotropic_E_nu(100, 0.25);
    SymSymR4 I_unbatched = SymSymR4::init_identity_sym();
    SymSymR4 C_batched = C_unbatched.batch_expand(2);
    SymSymR4 I_batched = I_unbatched.batch_expand(2);
    SECTION("unbatched,unbatched")
    {
      SymSymR4 result = C_unbatched * I_unbatched;
      REQUIRE(torch::allclose(result, C_unbatched));
    }
    SECTION("unbatched,batched")
    {
      SymSymR4 result = C_unbatched * I_batched;
      REQUIRE(torch::allclose(result, C_batched));
    }
    SECTION("batched,unbatched")
    {
      SymSymR4 result = C_batched * I_unbatched;
      REQUIRE(torch::allclose(result, C_batched));
    }
    SECTION("batched,batched")
    {
      SymSymR4 result = C_batched * I_batched;
      REQUIRE(torch::allclose(result, C_batched));
    }
  }
}
