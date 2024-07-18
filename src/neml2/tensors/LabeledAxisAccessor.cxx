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
#include "neml2/misc/error.h"

namespace neml2
{
LabeledAxisAccessor::operator std::vector<std::string>() const
{
  std::vector<std::string> v(_item_names.begin(), _item_names.end());
  return v;
}

bool
LabeledAxisAccessor::empty() const
{
  return _item_names.empty();
}

size_t
LabeledAxisAccessor::size() const
{
  return _item_names.size();
}

LabeledAxisAccessor
LabeledAxisAccessor::with_suffix(const std::string & suffix) const
{
  auto new_names = _item_names;
  new_names.back() += suffix;
  return new_names;
}

LabeledAxisAccessor
LabeledAxisAccessor::append(const LabeledAxisAccessor & axis) const
{
  return axis.prepend(*this);
}

LabeledAxisAccessor
LabeledAxisAccessor::prepend(const LabeledAxisAccessor & axis) const
{
  auto new_names = axis._item_names;
  new_names.insert(new_names.end(), _item_names.begin(), _item_names.end());
  return new_names;
}

LabeledAxisAccessor
LabeledAxisAccessor::slice(size_t n) const
{
  c10::SmallVector<std::string> new_names(_item_names.begin() + n, _item_names.end());
  return new_names;
}

LabeledAxisAccessor
LabeledAxisAccessor::slice(size_t n1, size_t n2) const
{
  c10::SmallVector<std::string> new_names(_item_names.begin() + n1, _item_names.begin() + n2);
  return new_names;
}

bool
LabeledAxisAccessor::start_with(const LabeledAxisAccessor & axis) const
{
  return slice(0, axis.size()) == axis;
}

void
LabeledAxisAccessor::validate_item_name(const std::string & name) const
{
  const auto x = name.find_first_of(" .,;/\t\n\v\f\r");
  neml_assert(x == std::string::npos,
              "Invalid item name: ",
              name,
              ". The item names cannot contain whitespace, '.', ',', ';', or '/'.");
}

bool
operator!=(const LabeledAxisAccessor & a, const LabeledAxisAccessor & b)
{
  return a.vec() != b.vec();
}

bool
operator==(const LabeledAxisAccessor & a, const LabeledAxisAccessor & b)
{
  return a.vec() == b.vec();
}

bool
operator<(const LabeledAxisAccessor & a, const LabeledAxisAccessor & b)
{
  return a.vec() < b.vec();
}

std::ostream &
operator<<(std::ostream & os, const LabeledAxisAccessor & accessor)
{
  for (size_t i = 0; i < accessor.vec().size(); i++)
  {
    if (i != 0)
      os << "/";
    os << accessor.vec()[i];
  }
  return os;
}
} // namespace neml2
