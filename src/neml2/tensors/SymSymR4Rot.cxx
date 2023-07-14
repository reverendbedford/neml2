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

#include "neml2/tensors/SymSymR4Rot.h"
#include "neml2/tensors/R4Rot.h"

namespace neml2
{

SymSymR4Rot
SymSymR4Rot::derivative(const Rotation & r, const SymSymR4 & T)
{
  auto full = R4Rot::derivative(r, T.to_full());

  // Someone please refactor me, but I can't figure it out now...
  auto res = torch::zeros({full.sizes()[0], 6, 6, 3}, TorchDefaults);

  for (TorchSize i = 0; i < 6; i++)
  {
    for (TorchSize j = 0; j < 6; j++)
    {
      res.index_put_({torch::indexing::Slice(), i, j, torch::indexing::Slice()},
                     full.index({torch::indexing::Slice(),
                                 utils::mandel_index[i][0],
                                 utils::mandel_index[i][1],
                                 utils::mandel_index[j][0],
                                 utils::mandel_index[j][1],
                                 torch::indexing::Slice()}) *
                         utils::mandelFactor(i) * utils::mandelFactor(j));
    }
  }

  return res;
}

} // namespace neml2
