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

#include "neml2/tensors/BatchTensor.h"

using namespace neml2;
using namespace torch::indexing;

TEST_CASE("BatchTensorBase", "[tensors]")
{
  torch::manual_seed(42);
  const auto & DTO = default_tensor_options();

  TorchShape B = {5, 3, 1, 2};             // batch shape
  TorchSize Bn = B.size();                 // batch dimension
  TorchShape D = {2, 5, 1, 3};             // base shape
  TorchSize Dn = D.size();                 // base dimension
  TorchSize n = Bn + Dn;                   // total dimension
  TorchShape BD = utils::add_shapes(B, D); // total shape
  TorchSize L = utils::storage_size(D);    // base storage

  SECTION("class BatchTensorBase")
  {
    SECTION("BatchTensorBase")
    {
      SECTION("default")
      {
        auto a = BatchTensor();
        REQUIRE(!a.defined());
      }

      SECTION("Tensor and batch dimension")
      {
        auto a = BatchTensor(torch::zeros(BD, DTO), B.size());
        REQUIRE(a.dim() == n);
        REQUIRE(a.batch_dim() == Bn);
        REQUIRE(a.base_dim() == Dn);
        REQUIRE(a.sizes() == BD);
        REQUIRE(a.batch_sizes() == B);
        REQUIRE(a.base_sizes() == D);
        REQUIRE(a.base_storage() == L);
      }

      SECTION("copy")
      {
        auto a = BatchTensor(torch::zeros(BD, DTO), B.size());
        auto b = BatchTensor(a);
        REQUIRE(b.dim() == a.dim());
        REQUIRE(b.batch_dim() == a.batch_dim());
        REQUIRE(b.base_dim() == a.base_dim());
        REQUIRE(b.sizes() == a.sizes());
        REQUIRE(b.batch_sizes() == a.batch_sizes());
        REQUIRE(b.base_sizes() == a.base_sizes());
        REQUIRE(b.base_storage() == a.base_storage());
      }
    }

    SECTION("empty")
    {
      SECTION("unbatched")
      {
        auto a = BatchTensor::empty(D, DTO);
        REQUIRE(!a.batched());
        REQUIRE(a.dim() == Dn);
        REQUIRE(a.batch_dim() == 0);
        REQUIRE(a.base_dim() == Dn);
        REQUIRE(a.sizes() == D);
        REQUIRE(a.batch_sizes() == TorchShape{});
        REQUIRE(a.base_sizes() == D);
        REQUIRE(a.base_storage() == L);
      }

      SECTION("batched")
      {
        auto a = BatchTensor::empty(B, D, DTO);
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

    SECTION("empty_like")
    {
      auto a = BatchTensor::empty(B, D, DTO);
      auto b = BatchTensor::empty_like(a);
      REQUIRE(b.dim() == a.dim());
      REQUIRE(b.batch_dim() == a.batch_dim());
      REQUIRE(b.base_dim() == a.base_dim());
      REQUIRE(b.sizes() == a.sizes());
      REQUIRE(b.batch_sizes() == a.batch_sizes());
      REQUIRE(b.base_sizes() == a.base_sizes());
      REQUIRE(b.base_storage() == a.base_storage());
    }

    SECTION("zeros")
    {
      SECTION("unbatched")
      {
        auto a = BatchTensor::zeros(D, DTO);
        REQUIRE(!a.batched());
        REQUIRE(a.dim() == Dn);
        REQUIRE(a.batch_dim() == 0);
        REQUIRE(a.base_dim() == Dn);
        REQUIRE(a.sizes() == D);
        REQUIRE(a.batch_sizes() == TorchShape{});
        REQUIRE(a.base_sizes() == D);
        REQUIRE(a.base_storage() == L);
        REQUIRE(torch::allclose(a, torch::zeros_like(a)));
      }

      SECTION("batched")
      {
        auto a = BatchTensor::zeros(B, D, DTO);
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

    SECTION("zeros_like")
    {
      auto a = BatchTensor::empty(B, D, DTO);
      auto b = BatchTensor::zeros_like(a);
      REQUIRE(b.dim() == a.dim());
      REQUIRE(b.batch_dim() == a.batch_dim());
      REQUIRE(b.base_dim() == a.base_dim());
      REQUIRE(b.sizes() == a.sizes());
      REQUIRE(b.batch_sizes() == a.batch_sizes());
      REQUIRE(b.base_sizes() == a.base_sizes());
      REQUIRE(b.base_storage() == a.base_storage());
      REQUIRE(torch::allclose(b, torch::zeros_like(b)));
    }

    SECTION("ones")
    {
      SECTION("unbatched")
      {
        auto a = BatchTensor::ones(D, DTO);
        REQUIRE(!a.batched());
        REQUIRE(a.dim() == Dn);
        REQUIRE(a.batch_dim() == 0);
        REQUIRE(a.base_dim() == Dn);
        REQUIRE(a.sizes() == D);
        REQUIRE(a.batch_sizes() == TorchShape{});
        REQUIRE(a.base_sizes() == D);
        REQUIRE(a.base_storage() == L);
        REQUIRE(torch::allclose(a, torch::ones_like(a)));
      }

      SECTION("batched")
      {
        auto a = BatchTensor::ones(B, D, DTO);
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

    SECTION("ones_like")
    {
      auto a = BatchTensor::empty(B, D, DTO);
      auto b = BatchTensor::ones_like(a);
      REQUIRE(b.dim() == a.dim());
      REQUIRE(b.batch_dim() == a.batch_dim());
      REQUIRE(b.base_dim() == a.base_dim());
      REQUIRE(b.sizes() == a.sizes());
      REQUIRE(b.batch_sizes() == a.batch_sizes());
      REQUIRE(b.base_sizes() == a.base_sizes());
      REQUIRE(b.base_storage() == a.base_storage());
      REQUIRE(torch::allclose(b, torch::ones_like(b)));
    }

    SECTION("full")
    {
      SECTION("unbatched")
      {
        Real init = 4.3;
        auto a = BatchTensor::full(D, init, DTO);
        REQUIRE(!a.batched());
        REQUIRE(a.dim() == Dn);
        REQUIRE(a.batch_dim() == 0);
        REQUIRE(a.base_dim() == Dn);
        REQUIRE(a.sizes() == D);
        REQUIRE(a.batch_sizes() == TorchShape{});
        REQUIRE(a.base_sizes() == D);
        REQUIRE(a.base_storage() == L);
        REQUIRE(torch::allclose(a, torch::full_like(a, init)));
      }

      SECTION("batched")
      {
        Real init = 2999;
        auto a = BatchTensor::full(B, D, init, DTO);
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

    SECTION("full_like")
    {
      Real init = -3.2;
      auto a = BatchTensor::empty(B, D, DTO);
      auto b = BatchTensor::full_like(a, init);
      REQUIRE(b.dim() == a.dim());
      REQUIRE(b.batch_dim() == a.batch_dim());
      REQUIRE(b.base_dim() == a.base_dim());
      REQUIRE(b.sizes() == a.sizes());
      REQUIRE(b.batch_sizes() == a.batch_sizes());
      REQUIRE(b.base_sizes() == a.base_sizes());
      REQUIRE(b.base_storage() == a.base_storage());
      REQUIRE(torch::allclose(b, init * torch::ones_like(b)));
    }

    SECTION("identity")
    {
      SECTION("unbatched")
      {
        TorchSize n = 21;
        TorchShape Deye = {n, n};
        auto a = BatchTensor::identity(n, DTO);
        REQUIRE(!a.batched());
        REQUIRE(a.dim() == 2);
        REQUIRE(a.batch_dim() == 0);
        REQUIRE(a.base_dim() == 2);
        REQUIRE(a.sizes() == Deye);
        REQUIRE(a.batch_sizes() == TorchShape{});
        REQUIRE(a.base_sizes() == Deye);
        REQUIRE(a.base_storage() == n * n);
        REQUIRE(torch::allclose(a, torch::eye(n, DTO)));
      }

      SECTION("batched")
      {
        TorchSize n = 33;
        TorchShape Deye = {n, n};
        TorchShape BDeye = utils::add_shapes(B, Deye);
        auto a = BatchTensor::identity(B, n, DTO);
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

    SECTION("linspace")
    {
      TorchSize nstep = 101;
      TorchSize dim = 2;
      TorchShape B_new = B;
      B_new.insert(B_new.begin() + dim, nstep);
      TorchShape BD_new = utils::add_shapes(B_new, D);

      auto a = BatchTensor::full(B, D, -5.5, DTO);
      auto b = BatchTensor::full(B, D, 123, DTO);
      auto c = BatchTensor::linspace(a, b, nstep, dim);
      REQUIRE(c.dim() == n + 1);
      REQUIRE(c.batch_dim() == Bn + 1);
      REQUIRE(c.base_dim() == Dn);
      REQUIRE(c.sizes() == BD_new);
      REQUIRE(c.batch_sizes() == B_new);
      REQUIRE(c.base_sizes() == D);
      REQUIRE(c.base_storage() == L);
    }

    SECTION("batch_index")
    {
      auto a = torch::rand({5, 3, 2, 1, 3}, DTO);
      auto b = BatchTensor(a, 3);
      TorchSlice i1 = {0};
      TorchSlice i2 = {2, 1};
      TorchSlice i3 = {Slice(), Slice(0, 2), 1};
      TorchSlice i4a = {2, Ellipsis, Slice(), Slice()};
      TorchSlice i4b = {2, Ellipsis};
      REQUIRE(torch::allclose(a.index(i1), b.batch_index(i1)));
      REQUIRE(torch::allclose(a.index(i2), b.batch_index(i2)));
      REQUIRE(torch::allclose(a.index(i3), b.batch_index(i3)));
      REQUIRE(torch::allclose(a.index(i4a), b.batch_index(i4b)));
    }

    SECTION("base_index")
    {
      auto a = torch::rand({5, 3, 2, 1, 3}, DTO);
      auto b = BatchTensor(a, 3);
      TorchSlice i1a = {Slice(), Slice(), Slice(), 0};
      TorchSlice i1b = {0};
      TorchSlice i2a = {Slice(), Slice(), Slice(), Ellipsis};
      TorchSlice i2b = {Ellipsis};
      TorchSlice i3a = {Ellipsis};
      TorchSlice i3b = {Ellipsis};
      TorchSlice i4a = {Slice(), Slice(), Slice(), Slice(), Slice(1, None)};
      TorchSlice i4b = {Slice(), Slice(1, None)};
      REQUIRE(torch::allclose(a.index(i1a), b.base_index(i1b)));
      REQUIRE(torch::allclose(a.index(i2a), b.base_index(i2b)));
      REQUIRE(torch::allclose(a.index(i3a), b.base_index(i3b)));
      REQUIRE(torch::allclose(a.index(i4a), b.base_index(i4b)));
    }

    SECTION("batch_index_put")
    {
      auto a = torch::rand({5, 3, 2, 1, 3}, DTO);
      auto b = BatchTensor(a, 3);
      auto c = torch::rand({3, 2, 1, 3}, DTO);
      TorchSlice ia = {2, Ellipsis, Slice(), Slice()};
      TorchSlice ib = {2, Ellipsis};
      b.batch_index_put(ib, c);
      REQUIRE(torch::allclose(a.index(ia), c));
    }

    SECTION("base_index_put")
    {
      auto a = torch::rand({5, 3, 2, 1, 3}, DTO);
      auto b = BatchTensor(a, 3);
      auto c = torch::rand({5, 3, 2, 1, 2}, DTO);
      TorchSlice ia = {Slice(), Slice(), Slice(), Slice(), Slice(1, None)};
      TorchSlice ib = {Slice(), Slice(1, None)};
      b.base_index_put(ib, c);
      REQUIRE(torch::allclose(a.index(ia), c));
    }

    SECTION("batch_expand")
    {
      TorchShape s0 = {5, 1, 2, 1, 5};
      TorchShape s = {5, 8, 2, 2, 5};
      auto a = BatchTensor::full(s0, {3, 3}, 5.25324, DTO);
      auto b = a.batch_expand(s);
      REQUIRE(b.storage().data_ptr() == a.storage().data_ptr());
      REQUIRE(b.batch_sizes() == s);
      REQUIRE(b.base_sizes() == a.base_sizes());
      REQUIRE(torch::sum(a - b).item<Real>() == Approx(0));
    }

    SECTION("base_expand")
    {
      TorchShape s0 = {2, 1, 3, 1, 3};
      TorchShape s = {2, 7, 3, 1, 3};
      auto a = BatchTensor::full({5, 1, 5}, s0, 1.32145, DTO);
      auto b = a.base_expand(s);
      REQUIRE(b.storage().data_ptr() == a.storage().data_ptr());
      REQUIRE(b.batch_sizes() == a.batch_sizes());
      REQUIRE(b.base_sizes() == s);
      // This is fun, as a and b are NOT broadcastable based on our broadcasting rules for batched
      // tensors because they have different base shapes. However, they _should_ be broadcastable
      // based on libTorch's original broadcasting rules. So we need to interpret them as
      // torch::Tensors first before we can compute a - b. This is the correct behavior.
      REQUIRE(torch::sum(torch::Tensor(a) - torch::Tensor(b)).item<Real>() == Approx(0));
    }

    SECTION("batch_expand_as")
    {
      TorchShape s0 = {5, 1, 2, 1, 5};
      TorchShape s = {5, 8, 2, 2, 5};
      auto a = BatchTensor::full(s0, {3, 3}, 5.25324, DTO);
      auto b = BatchTensor::full(s, {5}, 3.33, DTO); // base shapes can differ!
      auto c = a.batch_expand_as(b);
      REQUIRE(c.batch_sizes() == s);
      REQUIRE(c.base_sizes() == a.base_sizes());
    }

    SECTION("base_expand_as")
    {
      TorchShape s0 = {2, 1, 3, 1, 3};
      TorchShape s = {2, 7, 3, 1, 3};
      auto a = BatchTensor::full({5, 1, 5}, s0, 1.32145, DTO);
      auto b = BatchTensor::full({3, 2, 1}, s, 3.33, DTO); // batch shapes can differ!
      auto c = a.base_expand_as(b);
      REQUIRE(c.batch_sizes() == a.batch_sizes());
      REQUIRE(c.base_sizes() == s);
    }

    SECTION("batch_expand_copy")
    {
      TorchShape s0 = {5, 1, 2, 1, 5};
      TorchShape s = {5, 8, 2, 2, 5};
      auto a = BatchTensor::full(s0, {3, 3}, 5.25324, DTO);
      auto b = a.batch_expand_copy(s);
      REQUIRE(b.batch_sizes() == s);
      REQUIRE(b.base_sizes() == a.base_sizes());
      REQUIRE(torch::sum(a - b).item<Real>() == Approx(0));
    }

    SECTION("base_expand_copy")
    {
      TorchShape s0 = {2, 1, 3, 1, 3};
      TorchShape s = {2, 7, 3, 1, 3};
      auto a = BatchTensor::full({5, 1, 5}, s0, 1.32145, DTO);
      auto b = a.base_expand_copy(s);
      REQUIRE(b.batch_sizes() == a.batch_sizes());
      REQUIRE(b.base_sizes() == s);
      // This is fun, as a and b are NOT broadcastable based on our broadcasting rules for batched
      // tensors because they have different base shapes. However, they _should_ be broadcastable
      // based on libTorch's original broadcasting rules. So we need to interpret them as
      // torch::Tensors first before we can compute a - b. This is the correct behavior.
      REQUIRE(torch::sum(torch::Tensor(a) - torch::Tensor(b)).item<Real>() == Approx(0));
    }

    SECTION("batch_unsqueeze")
    {
      auto a = BatchTensor::full({2, 5}, {3, 3}, 5.25324, DTO);

      auto a1 = a.batch_unsqueeze(0);
      REQUIRE(a1.batch_sizes() == TorchShape{1, 2, 5});
      REQUIRE(a1.base_sizes() == a.base_sizes());

      auto a2 = a.batch_unsqueeze(1);
      REQUIRE(a2.batch_sizes() == TorchShape{2, 1, 5});
      REQUIRE(a2.base_sizes() == a.base_sizes());

      auto a3 = a.batch_unsqueeze(-2);
      REQUIRE(a3.batch_sizes() == TorchShape{2, 1, 5});
      REQUIRE(a3.base_sizes() == a.base_sizes());
    }

    SECTION("base_unsqueeze")
    {
      auto a = BatchTensor::full({2, 5}, {3, 3}, 5.25324, DTO);

      auto a1 = a.base_unsqueeze(0);
      REQUIRE(a1.batch_sizes() == a.batch_sizes());
      REQUIRE(a1.base_sizes() == TorchShape{1, 3, 3});

      auto a2 = a.base_unsqueeze(1);
      REQUIRE(a2.batch_sizes() == a.batch_sizes());
      REQUIRE(a2.base_sizes() == TorchShape{3, 1, 3});

      auto a3 = a.base_unsqueeze(-2);
      REQUIRE(a3.batch_sizes() == a.batch_sizes());
      REQUIRE(a3.base_sizes() == TorchShape{3, 1, 3});
    }

    SECTION("batch_transpose")
    {
      auto a = BatchTensor::full({2, 3, 5, 2}, {3, 3}, 5.25324, DTO);
      auto b = a.batch_transpose(1, 3);
      REQUIRE(b.batch_sizes() == TorchShape{2, 2, 5, 3});
      REQUIRE(b.base_sizes() == a.base_sizes());
    }

    SECTION("base_transpose")
    {
      auto a = BatchTensor::full({3, 3}, {5, 3, 5, 2}, 5.25324, DTO);
      auto b = a.base_transpose(0, 3);
      REQUIRE(b.batch_sizes() == a.batch_sizes());
      REQUIRE(b.base_sizes() == TorchShape{2, 3, 5, 5});
    }
  }

  SECTION("operator+")
  {
    auto a = BatchTensor(torch::tensor({{3.1, 2.2}, {2.2, -1.1}}, DTO), 0);
    auto b = BatchTensor::full({}, {2, 2}, 2.0, DTO);
    auto c = BatchTensor(torch::tensor({{5.1, 4.2}, {4.2, 0.9}}, DTO), 0);
    REQUIRE(torch::allclose(a + 2.0, c));
    REQUIRE(torch::allclose(a.batch_expand(B) + 2.0, c.batch_expand(B)));
    REQUIRE(torch::allclose(2.0 + a, c));
    REQUIRE(torch::allclose(2.0 + a.batch_expand(B), c.batch_expand(B)));
    REQUIRE(torch::allclose(a + b, c));
    REQUIRE(torch::allclose(a + b.batch_expand(B), c.batch_expand(B)));
    REQUIRE(torch::allclose(a.batch_expand(B) + b, c.batch_expand(B)));
    REQUIRE(torch::allclose(a.batch_expand(B) + b.batch_expand(B), c.batch_expand(B)));
  }

  SECTION("operator-")
  {
    auto a = BatchTensor(torch::tensor({{3.1, 2.2}, {2.2, -1.1}}, DTO), 0);
    auto b = BatchTensor::full({}, {2, 2}, 2.0, DTO);
    auto c = BatchTensor(torch::tensor({{1.1, 0.2}, {0.2, -3.1}}, DTO), 0);
    REQUIRE(torch::allclose(a - 2.0, c));
    REQUIRE(torch::allclose(a.batch_expand(B) - 2.0, c.batch_expand(B)));
    REQUIRE(torch::allclose(2.0 - a, -c));
    REQUIRE(torch::allclose(2.0 - a.batch_expand(B), -c.batch_expand(B)));
    REQUIRE(torch::allclose(a - b, c));
    REQUIRE(torch::allclose(a - b.batch_expand(B), c.batch_expand(B)));
    REQUIRE(torch::allclose(a.batch_expand(B) - b, c.batch_expand(B)));
    REQUIRE(torch::allclose(a.batch_expand(B) - b.batch_expand(B), c.batch_expand(B)));
  }

  SECTION("operator*")
  {
    auto a = BatchTensor(torch::tensor({{3.1, 2.2}, {2.2, -1.1}}, DTO), 0);
    auto b = BatchTensor::full({}, {2, 2}, 2.0, DTO);
    auto c = BatchTensor(torch::tensor({{6.2, 4.4}, {4.4, -2.2}}, DTO), 0);
    REQUIRE(torch::allclose(a * 2.0, c));
    REQUIRE(torch::allclose(a.batch_expand(B) * 2.0, c.batch_expand(B)));
    REQUIRE(torch::allclose(2.0 * a, c));
    REQUIRE(torch::allclose(2.0 * a.batch_expand(B), c.batch_expand(B)));
  }

  SECTION("operator/")
  {
    auto a = BatchTensor(torch::tensor({{3.1, 2.2}, {2.2, -1.1}}, DTO), 0);
    auto b = BatchTensor::full({}, {2, 2}, 2.0, DTO);
    auto c = BatchTensor(torch::tensor({{1.55, 1.1}, {1.1, -0.55}}, DTO), 0);
    auto cinv = BatchTensor(1.0 / torch::tensor({{1.55, 1.1}, {1.1, -0.55}}, DTO), 0);
    REQUIRE(torch::allclose(a / 2.0, c));
    REQUIRE(torch::allclose(a.batch_expand(B) / 2.0, c.batch_expand(B)));
    REQUIRE(torch::allclose(2.0 / a, cinv));
    REQUIRE(torch::allclose(2.0 / a.batch_expand(B), cinv.batch_expand(B)));
    REQUIRE(torch::allclose(a / b, c));
    REQUIRE(torch::allclose(a / b.batch_expand(B), c.batch_expand(B)));
    REQUIRE(torch::allclose(a.batch_expand(B) / b, c.batch_expand(B)));
    REQUIRE(torch::allclose(a.batch_expand(B) / b.batch_expand(B), c.batch_expand(B)));
  }

  SECTION("pow")
  {
    auto a = BatchTensor(torch::tensor({{3.0, 2.0}, {2.0, -1.1}}, DTO), 0);
    auto b = BatchTensor::full({}, {2, 2}, 2.0, DTO);
    auto c = BatchTensor(torch::tensor({{9.0, 4.0}, {4.0, 1.21}}, DTO), 0);
    REQUIRE(torch::allclose(math::pow(a, 2.0), c));
    REQUIRE(torch::allclose(math::pow(a.batch_expand(B), 2.0), c.batch_expand(B)));
    REQUIRE(torch::allclose(math::pow(a, b), c));
    REQUIRE(torch::allclose(math::pow(a.batch_expand(B), b), c.batch_expand(B)));
    REQUIRE(torch::allclose(math::pow(a, b.batch_expand(B)), c.batch_expand(B)));
    REQUIRE(torch::allclose(math::pow(a.batch_expand(B), b.batch_expand(B)), c.batch_expand(B)));
  }

  SECTION("sign")
  {
    auto a = BatchTensor(torch::tensor({{3.0, 2.0}, {2.0, -1.1}}, DTO), 0);
    auto b = BatchTensor(torch::tensor({{1.0, 1.0}, {1.0, -1.0}}, DTO), 0);
    REQUIRE(torch::allclose(math::sign(a), b));
    REQUIRE(torch::allclose(math::sign(a.batch_expand(B)), b.batch_expand(B)));
  }

  SECTION("heaviside")
  {
    auto a = BatchTensor(torch::tensor({{3.0, 2.0}, {2.0, -1.1}}, DTO), 0);
    auto b = BatchTensor(torch::tensor({{1.0, 1.0}, {1.0, 0.0}}, DTO), 0);
    REQUIRE(torch::allclose(math::heaviside(a), b));
    REQUIRE(torch::allclose(math::heaviside(a.batch_expand(B)), b.batch_expand(B)));
  }

  SECTION("macaulay")
  {
    auto a = BatchTensor(torch::tensor({{3.0, 2.0}, {2.0, -1.1}}, DTO), 0);
    auto b = BatchTensor(torch::tensor({{3.0, 2.0}, {2.0, 0.0}}, DTO), 0);
    REQUIRE(torch::allclose(math::macaulay(a), b));
    REQUIRE(torch::allclose(math::macaulay(a.batch_expand(B)), b.batch_expand(B)));
  }

  SECTION("dmacaulay")
  {
    auto a = BatchTensor(torch::tensor({{3.0, 2.0}, {2.0, -1.1}}, DTO), 0);
    auto b = BatchTensor(torch::tensor({{1.0, 1.0}, {1.0, 0.0}}, DTO), 0);
    REQUIRE(torch::allclose(math::dmacaulay(a), b));
    REQUIRE(torch::allclose(math::dmacaulay(a.batch_expand(B)), b.batch_expand(B)));
  }

  SECTION("sqrt")
  {
    auto a = BatchTensor(torch::tensor({{4.0, 9.0}, {25.0, 64.0}}, DTO), 0);
    auto b = BatchTensor(torch::tensor({{2.0, 3.0}, {5.0, 8.0}}, DTO), 0);
    REQUIRE(torch::allclose(math::sqrt(a), b));
    REQUIRE(torch::allclose(math::sqrt(a.batch_expand(B)), b.batch_expand(B)));
  }

  SECTION("exp")
  {
    auto a = BatchTensor(torch::tensor({{3.0, 2.0}, {2.0, -1.1}}, DTO), 0);
    auto b = BatchTensor(
        torch::tensor(
            {{20.085536923187668, 7.38905609893065}, {7.38905609893065, 0.33287108369807955}}, DTO),
        0);
    REQUIRE(torch::allclose(math::exp(a), b));
    REQUIRE(torch::allclose(math::exp(a.batch_expand(B)), b.batch_expand(B)));
  }

  SECTION("abs")
  {
    auto a = BatchTensor(torch::tensor({{3.0, 2.0}, {2.0, -1.1}}, DTO), 0);
    auto b = BatchTensor(torch::tensor({{3.0, 2.0}, {2.0, 1.1}}, DTO), 0);
    REQUIRE(torch::allclose(math::abs(a), b));
    REQUIRE(torch::allclose(math::abs(a.batch_expand(B)), b.batch_expand(B)));
  }
}
