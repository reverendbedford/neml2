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

#include "neml2/tensors/Tensor.h"

using namespace neml2;

TEST_CASE("Tensor", "[tensors]")
{
  const auto & DTO = default_tensor_options();

  TensorShape B = {5, 3, 1, 2};             // batch shape
  Size Bn = B.size();                       // batch dimension
  TensorShape D = {2, 5, 1, 3};             // base shape
  Size Dn = D.size();                       // base dimension
  Size n = Bn + Dn;                         // total dimension
  TensorShape BD = utils::add_shapes(B, D); // total shape
  Size L = utils::storage_size(D);          // base storage

  SECTION("empty")
  {
    SECTION("unbatched")
    {
      auto a = Tensor::empty(D, DTO);
      REQUIRE(!a.batched());
      REQUIRE(a.dim() == Dn);
      REQUIRE(a.batch_dim() == 0);
      REQUIRE(a.base_dim() == Dn);
      REQUIRE(a.sizes() == D);
      REQUIRE(a.batch_sizes() == TensorShape{});
      REQUIRE(a.base_sizes() == D);
      REQUIRE(a.base_storage() == L);
    }

    SECTION("batched")
    {
      auto a = Tensor::empty(B, D, DTO);
      REQUIRE(a.batched());
      REQUIRE(a.dim() == n);
      REQUIRE(a.batch_dim() == Bn);
      REQUIRE(a.base_dim() == Dn);
      REQUIRE(a.sizes() == BD);
      REQUIRE(a.batch_sizes() == B);
      REQUIRE(a.base_sizes() == D);
      REQUIRE(a.base_storage() == L);
    }
  }

  SECTION("zeros")
  {
    SECTION("unbatched")
    {
      auto a = Tensor::zeros(D, DTO);
      REQUIRE(!a.batched());
      REQUIRE(a.dim() == Dn);
      REQUIRE(a.batch_dim() == 0);
      REQUIRE(a.base_dim() == Dn);
      REQUIRE(a.sizes() == D);
      REQUIRE(a.batch_sizes() == TensorShape{});
      REQUIRE(a.base_sizes() == D);
      REQUIRE(a.base_storage() == L);
      REQUIRE(torch::allclose(a, torch::zeros_like(a)));
    }

    SECTION("batched")
    {
      auto a = Tensor::zeros(B, D, DTO);
      REQUIRE(a.batched());
      REQUIRE(a.dim() == n);
      REQUIRE(a.batch_dim() == Bn);
      REQUIRE(a.base_dim() == Dn);
      REQUIRE(a.sizes() == BD);
      REQUIRE(a.batch_sizes() == B);
      REQUIRE(a.base_sizes() == D);
      REQUIRE(a.base_storage() == L);
      REQUIRE(torch::allclose(a, torch::zeros_like(a)));
    }
  }

  SECTION("ones")
  {
    SECTION("unbatched")
    {
      auto a = Tensor::ones(D, DTO);
      REQUIRE(!a.batched());
      REQUIRE(a.dim() == Dn);
      REQUIRE(a.batch_dim() == 0);
      REQUIRE(a.base_dim() == Dn);
      REQUIRE(a.sizes() == D);
      REQUIRE(a.batch_sizes() == TensorShape{});
      REQUIRE(a.base_sizes() == D);
      REQUIRE(a.base_storage() == L);
      REQUIRE(torch::allclose(a, torch::ones_like(a)));
    }

    SECTION("batched")
    {
      auto a = Tensor::ones(B, D, DTO);
      REQUIRE(a.batched());
      REQUIRE(a.dim() == n);
      REQUIRE(a.batch_dim() == Bn);
      REQUIRE(a.base_dim() == Dn);
      REQUIRE(a.sizes() == BD);
      REQUIRE(a.batch_sizes() == B);
      REQUIRE(a.base_sizes() == D);
      REQUIRE(a.base_storage() == L);
      REQUIRE(torch::allclose(a, torch::ones_like(a)));
    }
  }

  SECTION("full")
  {
    SECTION("unbatched")
    {
      Real init = 4.3;
      auto a = Tensor::full(D, init, DTO);
      REQUIRE(!a.batched());
      REQUIRE(a.dim() == Dn);
      REQUIRE(a.batch_dim() == 0);
      REQUIRE(a.base_dim() == Dn);
      REQUIRE(a.sizes() == D);
      REQUIRE(a.batch_sizes() == TensorShape{});
      REQUIRE(a.base_sizes() == D);
      REQUIRE(a.base_storage() == L);
      REQUIRE(torch::allclose(a, torch::full_like(a, init)));
    }

    SECTION("batched")
    {
      Real init = 2999;
      auto a = Tensor::full(B, D, init, DTO);
      REQUIRE(a.batched());
      REQUIRE(a.dim() == n);
      REQUIRE(a.batch_dim() == Bn);
      REQUIRE(a.base_dim() == Dn);
      REQUIRE(a.sizes() == BD);
      REQUIRE(a.batch_sizes() == B);
      REQUIRE(a.base_sizes() == D);
      REQUIRE(a.base_storage() == L);
      REQUIRE(torch::allclose(a, torch::full_like(a, init)));
    }
  }

  SECTION("identity")
  {
    SECTION("unbatched")
    {
      Size n = 21;
      TensorShape Deye = {n, n};
      auto a = Tensor::identity(n, DTO);
      REQUIRE(!a.batched());
      REQUIRE(a.dim() == 2);
      REQUIRE(a.batch_dim() == 0);
      REQUIRE(a.base_dim() == 2);
      REQUIRE(a.sizes() == Deye);
      REQUIRE(a.batch_sizes() == TensorShape{});
      REQUIRE(a.base_sizes() == Deye);
      REQUIRE(a.base_storage() == n * n);
      REQUIRE(torch::allclose(a, torch::eye(n, DTO)));
    }

    SECTION("batched")
    {
      Size n = 33;
      TensorShape Deye = {n, n};
      TensorShape BDeye = utils::add_shapes(B, Deye);
      auto a = Tensor::identity(B, n, DTO);
      REQUIRE(a.batched());
      REQUIRE(a.dim() == Bn + 2);
      REQUIRE(a.batch_dim() == Bn);
      REQUIRE(a.base_dim() == 2);
      REQUIRE(a.sizes() == BDeye);
      REQUIRE(a.batch_sizes() == B);
      REQUIRE(a.base_sizes() == Deye);
      REQUIRE(a.base_storage() == n * n);
      REQUIRE(torch::allclose(a, torch::eye(n, DTO).expand(utils::add_shapes(B, -1, -1))));
    }
  }

  SECTION("bmm")
  {
    // A has shape (1, 2; 3, 5)
    auto A =
        Tensor(torch::tensor({{{{8.7205e-01, 3.5226e-01, 2.3191e-01, 4.3637e-01, 7.5373e-01},
                                {1.0045e-01, 9.7506e-01, 3.2655e-01, 9.6713e-01, 7.6809e-01},
                                {7.1273e-01, 5.0373e-01, 1.3844e-01, 3.0433e-01, 8.3720e-01}},
                               {{2.8995e-01, 5.7084e-01, 4.2301e-01, 2.9878e-01, 3.3778e-04},
                                {2.0580e-02, 3.7664e-01, 9.9539e-01, 9.0972e-01, 9.9746e-01},
                                {7.7913e-01, 6.6575e-01, 2.3632e-01, 3.1841e-01, 3.6544e-01}}}},
                             DTO),
               2);
    // B has shape (1; 5, 2)
    auto B = Tensor(torch::tensor({{{0.7480, 0.2192},
                                    {0.0742, 0.5324},
                                    {0.6179, 0.0834},
                                    {0.6298, 0.0578},
                                    {0.0945, 0.8719}}},
                                  DTO),
                    1);
    // C = math::bmm(A, B) should have shape (1, 2; 3, 2)
    auto C = Tensor(torch::tensor({{{{1.1678, 1.0804}, {1.0310, 1.2939}, {0.9269, 1.1835}},
                                    {{0.7088, 0.4203}, {1.3256, 1.2103}, {1.0133, 0.8819}}}},
                                  DTO),
                    2);
    REQUIRE(torch::allclose(math::bmm(A, B), C, /*rtol=*/0, /*atol=*/1e-4));
  }
}
