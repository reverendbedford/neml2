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

#include "neml2/tensors/LabeledAxisAccessor.h"

namespace neml2
{
LabeledAxisAccessor::operator std::vector<std::string>() const { return item_names; }

bool
LabeledAxisAccessor::empty() const
{
  return item_names.empty();
}

LabeledAxisAccessor
LabeledAxisAccessor::with_suffix(const std::string & suffix) const
{
  auto new_names = item_names;
  new_names.back() += suffix;
  return {new_names};
}

LabeledAxisAccessor
LabeledAxisAccessor::on(const std::string & axis) const
{
  auto new_names = item_names;
  new_names.insert(new_names.begin(), axis);
  return {new_names};
}

LabeledAxisAccessor
LabeledAxisAccessor::on(const LabeledAxisAccessor & axis) const
{
  auto new_names = axis.item_names;
  new_names.insert(new_names.end(), item_names.begin(), item_names.end());
  return {new_names};
}

LabeledAxisAccessor
LabeledAxisAccessor::peel(size_t n) const
{
  auto new_names = item_names;
  new_names.erase(new_names.begin(), new_names.begin() + n);
  return {new_names};
}

bool
LabeledAxisAccessor::operator==(const LabeledAxisAccessor & other) const
{
  return item_names == other.item_names;
}

bool
LabeledAxisAccessor::operator<(const LabeledAxisAccessor & other) const
{
  return item_names < other.item_names;
}

std::ostream &
operator<<(std::ostream & os, const LabeledAxisAccessor & accessor)
{
  for (size_t i = 0; i < accessor.item_names.size(); i++)
  {
    if (i != 0)
      os << "/";
    os << accessor.item_names[i];
  }
  return os;
}
} // namespace neml2
