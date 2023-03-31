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

#include "neml2/tensors/UserFixedDimTensor.h"

namespace neml2
{
register_NEML2_object_alt(UserScalar, "Scalar");
register_NEML2_object_alt(UserSymR2, "SymR2");
register_NEML2_object_alt(UserSymSymR4, "SymSymR4");

template <typename T>
ParameterSet
UserFixedDimTensor<T>::expected_params()
{
  ParameterSet params = NEML2Object::expected_params();
  params.set<std::vector<Real>>("values");
  params.set<std::vector<TorchSize>>("batch_shape") = std::vector<TorchSize>{1};
  return params;
}

template <typename T>
UserFixedDimTensor<T>::UserFixedDimTensor(const ParameterSet & params)
  : NEML2Object(params),
    T(torch::tensor(params.get<std::vector<Real>>("values"), default_tensor_options)
          .reshape(
              utils::add_shapes(params.get<std::vector<TorchSize>>("batch_shape"), T::_base_sizes)))
{
}

template class UserFixedDimTensor<Scalar>;
template class UserFixedDimTensor<SymR2>;
template class UserFixedDimTensor<SymSymR4>;
} // namespace neml2
