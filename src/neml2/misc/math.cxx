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

#include "neml2/misc/math.h"
#include "neml2/misc/error.h"

namespace neml2
{
namespace math
{
BatchTensor
full_to_mandel(const BatchTensor & full, TorchSize dim)
{
  using namespace torch::indexing;

  auto batch_dim = full.batch_dim();
  auto starting_dim = batch_dim + dim;
  auto trailing_dim = full.dim() - starting_dim - 2; // 2 comes from the symmetric axes (3,3)
  auto starting_shape = full.sizes().slice(0, starting_dim);
  auto trailing_shape = full.sizes().slice(starting_dim + 2);

  TorchSlice net(starting_dim, None);
  net.push_back(Ellipsis);
  net.insert(net.end(), trailing_dim, None);
  auto map =
      full_to_mandel_map.index(net).expand(utils::add_shapes(starting_shape, 6, trailing_shape));
  auto factor = full_to_mandel_factor.to(full).index(net);

  return BatchTensor(
      factor * torch::gather(full.reshape(utils::add_shapes(starting_shape, 9, trailing_shape)),
                             starting_dim,
                             map),
      batch_dim);
}

BatchTensor
mandel_to_full(const BatchTensor & mandel, TorchSize dim)
{
  using namespace torch::indexing;

  auto batch_dim = mandel.batch_dim();
  auto starting_dim = batch_dim + dim;
  auto trailing_dim = mandel.dim() - starting_dim - 1; // There's only 1 axis using Mandel
  auto starting_shape = mandel.sizes().slice(0, starting_dim);
  auto trailing_shape = mandel.sizes().slice(starting_dim + 1);

  TorchSlice net(starting_dim, None);
  net.push_back(Ellipsis);
  net.insert(net.end(), trailing_dim, None);
  auto map =
      mandel_to_full_map.index(net).expand(utils::add_shapes(starting_shape, 9, trailing_shape));
  auto factor = mandel_to_full_factor.to(mandel).index(net);

  return BatchTensor((factor * torch::gather(mandel, starting_dim, map))
                         .reshape(utils::add_shapes(starting_shape, 3, 3, trailing_shape)),
                     batch_dim);
}

BatchTensor
jacrev(const BatchTensor & out, const BatchTensor & p)
{
  neml_assert(p.batch_dim() == 0 || out.batch_sizes() == p.batch_sizes(),
              "If the parameter is batched, its batch shape must be the same as the batch shape "
              "of the output. However, the batch shape of the parameter is ",
              p.batch_sizes(),
              ", and the batch shape of the output is ",
              out.batch_sizes());

  // flatten out to handle arbitrarily shaped output
  auto outf = BatchTensor(
      out.reshape(utils::add_shapes(out.batch_sizes(), utils::storage_size(out.base_sizes()))),
      out.batch_dim());

  neml_assert_dbg(outf.base_dim() == 1, "Flattened output must be flat.");

  auto doutf_dp = BatchTensor::empty(
      outf.batch_sizes(), utils::add_shapes(outf.base_sizes(), p.base_sizes()), outf.options());

  for (TorchSize i = 0; i < outf.base_sizes()[0]; i++)
  {
    auto G = BatchTensor::zeros_like(outf);
    G.index_put_({torch::indexing::Ellipsis, i}, 1.0);
    auto doutfi_dp = torch::autograd::grad({outf},
                                           {p},
                                           {G},
                                           /*retain_graph=*/true,
                                           /*create_graph=*/false,
                                           /*allow_unused=*/false)[0];
    if (doutfi_dp.defined())
      doutf_dp.base_index_put({i, torch::indexing::Ellipsis}, doutfi_dp);
  }

  // reshape the derivative back to the correct shape
  auto dout_dp = BatchTensor(
      doutf_dp.reshape(utils::add_shapes(out.batch_sizes(), out.base_sizes(), p.base_sizes())),
      out.batch_dim());

  // factor to account for broadcasting
  Real factor = p.batch_dim() == 0 ? utils::storage_size(out.batch_sizes()) : 1;

  return dout_dp / factor;
}
} // namespace math
} // namespace neml2
