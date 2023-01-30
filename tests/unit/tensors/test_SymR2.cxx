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

#include "neml2/tensors/SymR2.h"
#include "neml2/tensors/SymSymR4.h"

using namespace neml2;

TEST_CASE("SymR2", "[SymR2]")
{
  SECTION("construct from one scalar")
  {
    SECTION("unbatched")
    {
      Scalar a(2.3);
      SymR2 result = SymR2::init(a);
      SymR2 correct(torch::tensor({{2.3, 2.3, 2.3, 0.0, 0.0, 0.0}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched")
    {
      Scalar a(torch::tensor({{2.3}, {3.4}}, TorchDefaults));
      SymR2 result = SymR2::init(a);
      SymR2 correct(torch::tensor({{2.3, 2.3, 2.3, 0.0, 0.0, 0.0}, {3.4, 3.4, 3.4, 0.0, 0.0, 0.0}},
                                  TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
  }

  SECTION("construct from three scalars")
  {
    SECTION("unbatched")
    {
      Scalar a11(2.3);
      Scalar a22(-1.3);
      Scalar a33(5.6);
      SymR2 result = SymR2::init(a11, a22, a33);
      SymR2 correct(torch::tensor({{2.3, -1.3, 5.6, 0.0, 0.0, 0.0}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched")
    {
      Scalar a11(torch::tensor({{2.3}, {3.4}}, TorchDefaults));
      Scalar a22(torch::tensor({{-1.3}, {6.4}}, TorchDefaults));
      Scalar a33(torch::tensor({{5.6}, {-1.1}}, TorchDefaults));
      SymR2 result = SymR2::init(a11, a22, a33);
      SymR2 correct(torch::tensor(
          {{2.3, -1.3, 5.6, 0.0, 0.0, 0.0}, {3.4, 6.4, -1.1, 0.0, 0.0, 0.0}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
  }

  SECTION("construct from six scalars")
  {
    SECTION("unbatched")
    {
      Scalar a11(2.3);
      Scalar a22(-1.3);
      Scalar a33(5.6);
      Scalar a23(3.8);
      Scalar a13(1.1);
      Scalar a12(-9.1);
      SymR2 result = SymR2::init(a11, a22, a33, a23, a13, a12);
      SymR2 correct(torch::tensor(
          {{2.3, -1.3, 5.6, utils::sqrt2 * 3.8, utils::sqrt2 * 1.1, utils::sqrt2 * -9.1}},
          TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched")
    {
      Scalar a11(torch::tensor({{2.3}, {3.4}}, TorchDefaults));
      Scalar a22(torch::tensor({{-1.3}, {6.4}}, TorchDefaults));
      Scalar a33(torch::tensor({{5.6}, {-1.1}}, TorchDefaults));
      Scalar a23(torch::tensor({{3.8}, {1.2}}, TorchDefaults));
      Scalar a13(torch::tensor({{1.1}, {5.5}}, TorchDefaults));
      Scalar a12(torch::tensor({{-9.1}, {3.1}}, TorchDefaults));
      SymR2 result = SymR2::init(a11, a22, a33, a23, a13, a12);
      SymR2 correct(torch::tensor(
          {{2.3, -1.3, 5.6, utils::sqrt2 * 3.8, utils::sqrt2 * 1.1, utils::sqrt2 * -9.1},
           {3.4, 6.4, -1.1, utils::sqrt2 * 1.2, utils::sqrt2 * 5.5, utils::sqrt2 * 3.1}},
          TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
  }

  TorchSize nbatch = 2;
  SymR2 A_unbatched(torch::arange(0, 6, TorchDefaults).reshape({1, 6}));
  SymR2 B_unbatched(torch::arange(1, 7, TorchDefaults).reshape({1, 6}));
  SymR2 A_batched(torch::arange(0, nbatch * 6, TorchDefaults).reshape({nbatch, 6}));
  SymR2 B_batched(torch::arange(1, nbatch * 6 + 1, TorchDefaults).reshape({nbatch, 6}));

  SECTION("operator(i, j)")
  {
    SECTION("unbatched")
    {
      Scalar result = A_unbatched(1, 2);
      Scalar correct(3 / utils::sqrt2);
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched")
    {
      Scalar result = A_batched(1, 2);
      Scalar correct(torch::tensor({{3 / utils::sqrt2}, {9 / utils::sqrt2}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
  }

  SECTION("tr(A)")
  {
    SECTION("unbatched")
    {
      Scalar result = A_unbatched.tr();
      Scalar correct(3);
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched")
    {
      Scalar result = A_batched.tr();
      Scalar correct(torch::tensor({{3}, {21}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
  }

  SECTION("vol(A)")
  {
    SECTION("unbatched")
    {
      SymR2 result = A_unbatched.vol();
      SymR2 correct(torch::tensor({{1, 1, 1, 0, 0, 0}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched")
    {
      SymR2 result = A_batched.vol();
      SymR2 correct(torch::tensor({{1, 1, 1, 0, 0, 0}, {7, 7, 7, 0, 0, 0}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
  }

  SECTION("dev(A)")
  {
    SECTION("unbatched")
    {
      SymR2 result = A_unbatched.dev();
      SymR2 correct(torch::tensor({{-1, 0, 1, 3, 4, 5}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched")
    {
      SymR2 result = A_batched.dev();
      SymR2 correct(torch::tensor({{-1, 0, 1, 3, 4, 5}, {-1, 0, 1, 9, 10, 11}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
  }

  SECTION("det(A)")
  {
    SECTION("unbatched")
    {
      Scalar result = A_unbatched.det();
      Scalar correct(9.4264049530);
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched")
    {
      Scalar result = A_batched.det();
      Scalar correct(torch::tensor({{9.4264049530}, {-40.9642715454}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
  }

  SECTION("norm(A)^2")
  {
    SECTION("unbatched")
    {
      Scalar result = A_unbatched.norm_sq();
      Scalar correct(55);
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched")
    {
      Scalar result = A_batched.norm_sq();
      Scalar correct(torch::tensor({{55}, {451}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
  }

  SECTION("norm(A)")
  {
    SECTION("unbatched")
    {
      Scalar result = A_unbatched.norm();
      Scalar correct(std::sqrt(55));
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched")
    {
      Scalar result = A_batched.norm();
      Scalar correct(torch::tensor({{std::sqrt(55)}, {std::sqrt(451)}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
  }

  SECTION("ij,ij->")
  {
    SECTION("unbatched,unbatched")
    {
      Scalar result = A_unbatched.inner(B_unbatched);
      Scalar correct(70);
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("unbatched,batched")
    {
      Scalar result = A_unbatched.inner(B_batched);
      Scalar correct(torch::tensor({{70}, {160}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched,unbatched")
    {
      Scalar result = A_batched.inner(B_unbatched);
      Scalar correct(torch::tensor({{70}, {196}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched,batched")
    {
      Scalar result = A_batched.inner(B_batched);
      Scalar correct(torch::tensor({{70}, {502}}, TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
  }

  SECTION("ij,kl->ijkl")
  {
    SECTION("unbatched,unbatched")
    {
      SymSymR4 result = A_unbatched.outer(B_unbatched);
      SymSymR4 correct(torch::tensor({{{0, 0, 0, 0, 0, 0},
                                       {1, 2, 3, 4, 5, 6},
                                       {2, 4, 6, 8, 10, 12},
                                       {3, 6, 9, 12, 15, 18},
                                       {4, 8, 12, 16, 20, 24},
                                       {5, 10, 15, 20, 25, 30}}},
                                     TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched,batched")
    {
      SymSymR4 result = A_batched.outer(B_batched);
      SymSymR4 correct(torch::tensor({{{0, 0, 0, 0, 0, 0},
                                       {1, 2, 3, 4, 5, 6},
                                       {2, 4, 6, 8, 10, 12},
                                       {3, 6, 9, 12, 15, 18},
                                       {4, 8, 12, 16, 20, 24},
                                       {5, 10, 15, 20, 25, 30}},
                                      {{42, 48, 54, 60, 66, 72},
                                       {49, 56, 63, 70, 77, 84},
                                       {56, 64, 72, 80, 88, 96},
                                       {63, 72, 81, 90, 99, 108},
                                       {70, 80, 90, 100, 110, 120},
                                       {77, 88, 99, 110, 121, 132}}},
                                     TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
  }
}
