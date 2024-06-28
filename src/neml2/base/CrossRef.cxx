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

#include "neml2/base/CrossRef.h"
#include "neml2/base/Factory.h"
#include "neml2/misc/parser_utils.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/macros.h"

namespace neml2
{
template <>
CrossRef<torch::Tensor>::operator torch::Tensor() const
{
  try
  {
    // If it is just a number, we can still create a tensor out of it
    return torch::tensor(utils::parse<Real>(_raw_str), default_tensor_options());
  }
  catch (const ParserException & e1)
  {
    // Conversion to a number failed...
    try
    {
      // It could be a CSV...
      return utils::parse_csv(_raw_str, default_tensor_options());
    }
    catch (const ParserException & e2)
    {
      try
      {
        // Or it could be torch::Tensor defined under the [Tensors] section
        return Factory::get_object<torch::Tensor>("Tensors", _raw_str);
      }
      catch (const std::exception & e3)
      {
        throw NEMLException(
            "While resolving a cross reference '" + _raw_str +
            "', several attempts have been made but all failed.\n\nParsing it as a "
            "Real failed with error message:\n" +
            e1.what() + "\n\nParsing it as a CSV file failed with error message:\n" + e2.what() +
            "\n\nParsing it as the name of a torch::Tensor under the [Tensors] "
            "section failed with error message:\n" +
            e3.what());
      }
    }
  }
}

template <>
CrossRef<BatchTensor>::operator BatchTensor() const
{
  try
  {
    // If it is just a number, we can still create a tensor out of it
    return BatchTensor::full({}, {}, utils::parse<Real>(_raw_str), default_tensor_options());
  }
  catch (const ParserException & e1)
  {
    // Conversion to a number failed...
    try
    {
      // It could be BatchTensor defined under the [Tensors] section
      return Factory::get_object<BatchTensor>("Tensors", _raw_str);
    }
    catch (const std::exception & e2)
    {
      throw NEMLException("While resolving a cross reference '" + _raw_str +
                          "', several attempts have been made but all failed.\n\nParsing it as a "
                          "Real failed with error message:\n" +
                          e1.what() +
                          "\n\nParsing it as the name of a BatchTensor under the [Tensors] "
                          "section failed with error message:\n" +
                          e2.what());
    }
  }
}

template class CrossRef<torch::Tensor>;
#define CROSSREF_SPECIALIZE_FIXEDDIMTENSOR(T)                                                      \
  template <>                                                                                      \
  CrossRef<T>::operator T() const                                                                  \
  {                                                                                                \
    try                                                                                            \
    {                                                                                              \
      return T::full(utils::parse<Real>(_raw_str));                                                \
    }                                                                                              \
    catch (const ParserException & e1)                                                             \
    {                                                                                              \
      try                                                                                          \
      {                                                                                            \
        return T(utils::parse_csv(_raw_str, default_tensor_options()));                            \
      }                                                                                            \
      catch (const ParserException & e2)                                                           \
      {                                                                                            \
        try                                                                                        \
        {                                                                                          \
          return Factory::get_object<T>("Tensors", _raw_str);                                      \
        }                                                                                          \
        catch (const std::exception & e3)                                                          \
        {                                                                                          \
          throw NEMLException(                                                                     \
              "While resolving a cross reference '" + _raw_str +                                   \
              "', several attempts have been made but all failed.\n\nParsing it as a "             \
              "Real failed with error message:\n" +                                                \
              e1.what() + "\n\nParsing it as a CSV file failed with error message:\n" +            \
              e2.what() +                                                                          \
              "\n\nParsing it as the name of a tensor under the [Tensors] "                        \
              "section failed with error message:\n" +                                             \
              e3.what());                                                                          \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
  }                                                                                                \
  static_assert(true)

FOR_ALL_FIXEDDIMTENSOR(CROSSREF_SPECIALIZE_FIXEDDIMTENSOR);
} // namesace neml2
