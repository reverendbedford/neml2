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
LabeledAxisAccessor::LabeledAxisAccessor(const char * name)
  : _item_names({std::string(name)})
{
  validate_item_name(_item_names[0]);
}

LabeledAxisAccessor::LabeledAxisAccessor(const std::string & name)
  : _item_names({name})
{
  validate_item_name(name);
}

LabeledAxisAccessor::LabeledAxisAccessor(const std::vector<std::string> & names)
  : _item_names(names)
{
  for (const auto & name : names)
    validate_item_name(name);
}

LabeledAxisAccessor::LabeledAxisAccessor(const std::initializer_list<std::string> & names)
  : _item_names(names)
{
  for (const auto & name : _item_names)
    validate_item_name(name);
}

LabeledAxisAccessor::LabeledAxisAccessor(const LabeledAxisAccessor & other)
  : _item_names(other._item_names)
{
}

LabeledAxisAccessor &
LabeledAxisAccessor::operator=(const LabeledAxisAccessor & other)
{
  _item_names = other._item_names;
  return *this;
}

LabeledAxisAccessor::operator std::vector<std::string>() const { return _item_names; }

bool
LabeledAxisAccessor::empty() const
{
  return _item_names.empty();
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
  return axis.on(*this);
}

LabeledAxisAccessor
LabeledAxisAccessor::on(const LabeledAxisAccessor & axis) const
{
  auto new_names = axis._item_names;
  new_names.insert(new_names.end(), _item_names.begin(), _item_names.end());
  return new_names;
}

LabeledAxisAccessor
LabeledAxisAccessor::slice(size_t n) const
{
  auto new_names = _item_names;
  new_names.erase(new_names.begin(), new_names.begin() + n);
  return new_names;
}

LabeledAxisAccessor
LabeledAxisAccessor::slice(size_t n1, size_t n2) const
{
  auto new_names = _item_names;
  new_names.erase(new_names.begin() + n2, new_names.end());
  new_names.erase(new_names.begin(), new_names.begin() + n1);
  return new_names;
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
