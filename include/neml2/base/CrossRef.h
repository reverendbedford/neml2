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
 * @brief The wrapper (decorator) for cross-referencing unresolved values at parse time.
 *
 * All tokens in the input files are essentially strings, and it is not always possible to represent
 * all quantities as strings. For example, more sophisticated tensor manufacturing methods require
 * inputs like batch size, device type, etc., and cannot be specified with a single string literal.
 * In such scenarios, it is preferable to interpret the string literal as a "label"
 * cross-referencing another object. This wrapper provides cross-referencing capabilities to most
 * input file types. The object _name_ is used as the label, and the object value is not resolved
 * until implicit conversion occurs.
 *
 * @tparam T The final resolved type
 */
template <typename T>
class CrossRef
{
public:
  CrossRef() = default;

  /**
   * @brief Construct a new CrossRef object
   *
   * @param raw_str The raw string literal extracted from the input file
   */
  CrossRef(const std::string & raw_str)
    : _raw_str(raw_str)
  {
  }

  /**
   * @brief Assignment operator
   */
  CrossRef<T> & operator=(const std::string & other);

  /**
   * @brief Implicit conversion operator.
   *
   * It is assumed that the cross-referenced object has already been manufactured at this point.
   *
   * @return T The resolved value.
   */
  operator T() const;

  /**
   * @brief Get the raw string literal
   *
   * @return const std::string& The raw string literal.
   */
  const std::string & raw() const { return _raw_str; }

private:
  /// The raw string literal.
  std::string _raw_str;
};

template <typename T>
std::ostream &
operator<<(std::ostream & os, const CrossRef<T> & cr)
{
  os << cr.raw();
  return os;
}
} // namespace neml2
