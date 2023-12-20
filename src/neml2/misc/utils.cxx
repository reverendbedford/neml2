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

#include "neml2/misc/utils.h"

namespace neml2
{
namespace utils
{
TorchSize
storage_size(TorchShapeRef shape)
{
  TorchSize sz = 1;
  return std::accumulate(shape.begin(), shape.end(), sz, std::multiplies<TorchSize>());
}

TorchShape
pad_prepend(TorchShapeRef s, TorchSize dim, TorchSize pad)
{
  auto s2 = s.vec();
  s2.insert(s2.begin(), dim - s.size(), pad);
  return s2;
}

TorchShape
pad_append(TorchShapeRef s, TorchSize dim, TorchSize pad)
{
  auto s2 = s.vec();
  s2.insert(s2.end(), dim - s.size(), pad);
  return s2;
}

// LCOV_EXCL_START
std::string
indentation(int level, int indent)
{
  std::stringstream ss;
  std::string space(indent, ' ');
  for (int i = 0; i < level; i++)
    ss << space;
  return ss.str();
}
// LCOV_EXCL_STOP

namespace details
{
TorchShape
add_shapes_impl(TorchShape & net)
{
  return std::move(net);
}
} // namespace details
} // namespace utils
} // namespace neml2
