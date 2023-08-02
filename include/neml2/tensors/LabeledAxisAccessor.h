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

#include <cstddef>

namespace neml2
{
/**
The accessor containing all the information needed to access an item in a `LabeledAxis`. The
accessor consists of an arbitrary number of item names.
The last item name can be either a variable name or a sub-axis name.
All the other item names are considered to be sub-axis names.
*/
class LabeledAxisAccessor
{
public:
  std::vector<std::string> item_names;

  operator std::vector<std::string>() const;

  bool empty() const;

  LabeledAxisAccessor with_suffix(const std::string & suffix) const;

  LabeledAxisAccessor on(const std::string & axis) const;

  LabeledAxisAccessor on(const LabeledAxisAccessor & axis) const;

  LabeledAxisAccessor peel(size_t n = 1) const;

  bool operator==(const LabeledAxisAccessor & other) const;

  bool operator<(const LabeledAxisAccessor & other) const;

  friend std::ostream & operator<<(std::ostream & os, const LabeledAxisAccessor & accessor);
};

std::ostream & operator<<(std::ostream & os, const LabeledAxisAccessor & accessor);
} // namespace neml2
