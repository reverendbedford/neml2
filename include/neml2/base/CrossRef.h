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
#include <sstream>

#include "neml2/base/Factory.h"
#include "neml2/misc/parser_utils.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/macros.h"

namespace neml2
{
template <typename T>
class CrossRef;
template <typename T>
std::stringstream & operator>>(std::stringstream &, CrossRef<T> &);

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
   * @brief Assignment operator
   *
   * This simply assigns the string without parsing and resolving the cross-reference
   */
  CrossRef<T> & operator=(const std::string & other);

  /**
   * @brief Implicit conversion operator.
   *
   * The underlying string is parsed and used to resolve the cross-reference. It is assumed that the
   * cross-referenced object has already been manufactured at this point.
   */
  operator T() const;

  /// Test equality
  bool operator==(const CrossRef<T> & other) const { return _raw_str == other.raw(); }

  /**
   * @brief Get the raw string literal
   *
   * @return const std::string& The raw string literal.
   */
  const std::string & raw() const { return _raw_str; }

  friend std::stringstream & operator>> <>(std::stringstream & in, CrossRef<T> &);

private:
  /// The raw string literal.
  std::string _raw_str;
};
} // namespace neml2

///////////////////////////////////////////////////////////////////////////////
// Implementations
///////////////////////////////////////////////////////////////////////////////

namespace neml2
{
template <typename T>
CrossRef<T> &
CrossRef<T>::operator=(const std::string & other)
{
  _raw_str = other;
  return *this;
}

template <typename T>
std::ostream &
operator<<(std::ostream & os, const CrossRef<T> & cr)
{
  os << cr.raw();
  return os;
}

template <typename T>
std::stringstream &
operator>>(std::stringstream & ss, CrossRef<T> & cr)
{
  ss >> cr._raw_str;
  return ss;
}

template <>
inline CrossRef<torch::Tensor>::operator torch::Tensor() const
{
  try
  {
    // If it is just a number, we can still create a tensor out of it
    return torch::tensor(utils::parse<Real>(_raw_str), default_tensor_options());
  }
  catch (const ParserException & e)
  {
    // Conversion to a number failed, so it might be the name of another tensor
    return Factory::get_object<torch::Tensor>("Tensors", _raw_str);
  }
}

template <>
inline CrossRef<BatchTensor>::operator BatchTensor() const
{
  try
  {
    // If it is just a number, we can still create a Scalar out of it
    return BatchTensor::full({}, {}, utils::parse<Real>(_raw_str), default_tensor_options());
  }
  catch (const ParserException & e)
  {
    // Conversion to a number failed, so it might be the name of another BatchTensor
    return Factory::get_object<BatchTensor>("Tensors", _raw_str);
  }
}

template class CrossRef<torch::Tensor>;
#define CROSSREF_SPECIALIZE_FIXEDDIMTENSOR(T)                                                      \
  template <>                                                                                      \
  inline CrossRef<T>::operator T() const                                                           \
  {                                                                                                \
    try                                                                                            \
    {                                                                                              \
      return T::full(utils::parse<Real>(_raw_str));                                                \
    }                                                                                              \
    catch (const ParserException & e)                                                              \
    {                                                                                              \
      return Factory::get_object<T>("Tensors", _raw_str);                                          \
    }                                                                                              \
  }                                                                                                \
  static_assert(true)

FOR_ALL_FIXEDDIMTENSOR(CROSSREF_SPECIALIZE_FIXEDDIMTENSOR);
} // namespace neml2
