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


#pragma once

#include "neml2/misc/types.h"

namespace neml2
{
namespace utils
{
constexpr double sqrt2 = 1.4142135623730951;

inline constexpr double
mandelFactor(TorchSize i)
{
  return i < 3 ? 1.0 : sqrt2;
}

inline TorchSize
storage_size(const TorchShape & shape)
{
  TorchSize sz = 1;
  return std::accumulate(shape.begin(), shape.end(), sz, std::multiplies<TorchSize>());
}

template <typename... TorchShapeRef>
inline TorchShape
add_shapes(TorchShapeRef... shapes)
{
  TorchShape net;
  (net.insert(net.end(), shapes.begin(), shapes.end()), ...);
  return net;
}

std::string indentation(int level, int indent = 2);

template <typename T>
std::string
stringify(const T & t)
{
  std::ostringstream os;
  os << t;
  return os.str();
}
} // namespace utils
} // namespace neml2
