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

#include "neml2/drivers/ResultContainer.h"

namespace neml2
{
void
ResultContainer::emplace(const std::string & key, const LabeledVector & value)
{
  update({{key, std::make_shared<LabeledVectorContainer>(value)}});
}

namespace math
{
bool
allclose(const ResultContainer & a, const ResultContainer & b, Real rtol, Real atol)
{
  for (auto key : a.keys())
  {
    if (!b.contains(key))
      return false;
    auto a_value = a.at<LabeledVectorContainer>(key);
    auto b_value = b.at<LabeledVectorContainer>(key);
    if (!math::allclose(a_value, b_value, rtol, atol))
      return false;
  }
  return true;
}
} // namespace math
} // namespace neml2
