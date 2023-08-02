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

#include "neml2/tensors/LogSpaceTensor.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(LogSpaceTensor);

ParameterSet
LogSpaceTensor::expected_params()
{
  ParameterSet params = NEML2Object::expected_params();
  params.set<CrossRef<torch::Tensor>>("start") = "0";
  params.set<CrossRef<torch::Tensor>>("end");
  params.set<TorchSize>("steps");
  params.set<Real>("base") = 10;
  return params;
}

LogSpaceTensor::LogSpaceTensor(const ParameterSet & params)
  : NEML2Object(params),
    torch::Tensor(math::logspace(params.get<CrossRef<torch::Tensor>>("start"),
                                 params.get<CrossRef<torch::Tensor>>("end"),
                                 params.get<TorchSize>("steps"),
                                 params.get<Real>("base"),
                                 default_tensor_options))
{
}
} // namespace neml2
