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

#include <torch/torch.h>

#include "neml2/base/CrossRef.h"
#include "neml2/base/Factory.h"
#include "neml2/misc/parser_utils.h"
#include "neml2/tensors/Scalar.h"

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
  neml_assert(!_raw_str.empty(), "Trying to retrieve a torch::Tensor before it is being set.");
  try
  {
    // If it is just a number, we can still create a tensor out of it
    return torch::tensor(utils::parse<Real>(_raw_str));
  }
  catch (const ParserException & e)
  {
    // Conversion to a number failed, so it might be the name of another tensor
    return Factory::get_object<torch::Tensor>("Tensors", _raw_str);
  }
}

template <>
CrossRef<Scalar>::operator Scalar() const
{
  neml_assert(!_raw_str.empty(), "Trying to retrieve a Scalar before it is being set.");
  try
  {
    // If it is just a number, we can still create a Scalar out of it
    return Scalar(utils::parse<Real>(_raw_str));
  }
  catch (const ParserException & e)
  {
    // Conversion to a number failed, so it might be the name of another Scalar
    return Factory::get_object<Scalar>("Tensors", _raw_str);
  }
}

template <>
CrossRef<SymR2>::operator SymR2() const
{
  neml_assert(!_raw_str.empty(), "Trying to retrieve a SymR2 before it is being set.");
  try
  {
    // If it is just a number, we can still create a SymR2 out of it:
    // This will be a SymR2 with diagonals equal to the number.
    return SymR2::init(utils::parse<Real>(_raw_str));
  }
  catch (const ParserException & e)
  {
    // Conversion to a number failed, so it might be the name of another SymR2
    return Factory::get_object<SymR2>("Tensors", _raw_str);
  }
}

template class CrossRef<torch::Tensor>;
template class CrossRef<Scalar>;
template class CrossRef<SymR2>;
} // namesace neml2
