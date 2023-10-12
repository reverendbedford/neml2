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
#include "neml2/models/crystallography/SymmetryOperator.h"
#include "neml2/models/crystallography/MillerIndex.h"

#define specialize_crossref_FixedDimTensor(tensor_type)                                            \
  template <>                                                                                      \
  CrossRef<tensor_type>::operator tensor_type() const                                              \
  {                                                                                                \
    try                                                                                            \
    {                                                                                              \
      return tensor_type::full(utils::parse<Real>(_raw_str));                                      \
    }                                                                                              \
    catch (const ParserException & e)                                                              \
    {                                                                                              \
      return Factory::get_object<tensor_type>("Tensors", _raw_str);                                \
    }                                                                                              \
  }                                                                                                \
  template class CrossRef<tensor_type>

namespace neml2
{
template <typename T>
CrossRef<T> &
CrossRef<T>::operator=(const std::string & other)
{
  _raw_str = other;
  return *this;
}

template <>
CrossRef<torch::Tensor>::operator torch::Tensor() const
{
  try
  {
    // If it is just a number, we can still create a tensor out of it
    return torch::tensor(utils::parse<Real>(_raw_str), default_tensor_options);
  }
  catch (const ParserException & e)
  {
    // Conversion to a number failed, so it might be the name of another tensor
    return Factory::get_object<torch::Tensor>("Tensors", _raw_str);
  }
}

template <>
CrossRef<BatchTensor>::operator BatchTensor() const
{
  try
  {
    // If it is just a number, we can still create a Scalar out of it
    return BatchTensor::full({}, {}, utils::parse<Real>(_raw_str), default_tensor_options);
  }
  catch (const ParserException & e)
  {
    // Conversion to a number failed, so it might be the name of another BatchTensor
    return Factory::get_object<BatchTensor>("Tensors", _raw_str);
  }
}

template class CrossRef<torch::Tensor>;
template class CrossRef<BatchTensor>;

specialize_crossref_FixedDimTensor(Scalar);
specialize_crossref_FixedDimTensor(Vec);
specialize_crossref_FixedDimTensor(Rot);
specialize_crossref_FixedDimTensor(R2);
specialize_crossref_FixedDimTensor(SR2);
specialize_crossref_FixedDimTensor(R3);
specialize_crossref_FixedDimTensor(SFR3);
specialize_crossref_FixedDimTensor(R4);
specialize_crossref_FixedDimTensor(SSR4);
specialize_crossref_FixedDimTensor(R5);
specialize_crossref_FixedDimTensor(SSFR5);
specialize_crossref_FixedDimTensor(crystallography::SymmetryOperator);
specialize_crossref_FixedDimTensor(crystallography::MillerIndex);
} // namesace neml2
