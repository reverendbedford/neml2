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

#include "neml2/tensors/user_tensors/UserFixedDimTensor.h"

namespace neml2
{
#define USERFIXEDDIMTENSOR_REGISTER(T) register_NEML2_object_alias(User##T, #T)
FOR_ALL_FIXEDDIMTENSOR(USERFIXEDDIMTENSOR_REGISTER);

template <typename T>
OptionSet
UserFixedDimTensor<T>::expected_options()
{
  OptionSet options = UserTensor::expected_options();
  options.set<std::vector<Real>>("values");
  options.set<TorchShape>("batch_shape") = {};
  return options;
}

template <typename T>
UserFixedDimTensor<T>::UserFixedDimTensor(const OptionSet & options)
  : T(T::empty(options.get<TorchShape>("batch_shape"), default_tensor_options())),
    UserTensor(options)
{
  auto vals = options.get<std::vector<Real>>("values");
  auto flat = torch::tensor(vals, default_tensor_options());
  if (vals.size() == size_t(this->base_storage()))
    this->index_put_({torch::indexing::Ellipsis}, flat.reshape(this->base_sizes()));
  else if (vals.size() == size_t(utils::storage_size(this->sizes())))
    this->index_put_({torch::indexing::Ellipsis}, flat.reshape(this->sizes()));
  else
    neml_assert(false,
                "Number of values ",
                vals.size(),
                " must equal to either the base storage size ",
                this->base_storage(),
                " or the total storage size ",
                utils::storage_size(this->sizes()));
}

#define USERFIXEDDIMTENSOR_INSTANTIATE_FIXEDDIMTENSOR(T) template class UserFixedDimTensor<T>
FOR_ALL_FIXEDDIMTENSOR(USERFIXEDDIMTENSOR_INSTANTIATE_FIXEDDIMTENSOR);
} // namespace neml2
