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

#include "neml2/tensors/user_tensors/LinspaceBatchTensor.h"
#include "neml2/base/CrossRef.h"

namespace neml2
{
register_NEML2_object(LinspaceBatchTensor);

OptionSet
LinspaceBatchTensor::expected_options()
{
  OptionSet options = UserTensor::expected_options();
  options.doc() = "Construct a BatchTensor linearly spaced on the batch dimensions. See "
                  "neml2::BatchTensorBase::linspace for a detailed explanation.";

  options.set<CrossRef<BatchTensor>>("start");
  options.set("start").doc() = "The starting tensor";

  options.set<CrossRef<BatchTensor>>("end");
  options.set("end").doc() = "The ending tensor";

  options.set<Size>("nstep");
  options.set("nstep").doc() = "The number of steps with even spacing along the new dimension";

  options.set<Size>("dim") = 0;
  options.set("dim").doc() = "Where to insert the new dimension";

  options.set<Size>("batch_dim") = -1;
  options.set("batch_dim").doc() = "Batch dimension of the output";

  options.set<TensorShape>("batch_expand") = TensorShape();
  options.set("batch_expand").doc() = "After construction, perform an additional batch expanding "
                                      "operation into the given batch shape.";

  return options;
}

LinspaceBatchTensor::LinspaceBatchTensor(const OptionSet & options)
  : BatchTensor(BatchTensor::linspace(options.get<CrossRef<BatchTensor>>("start"),
                                      options.get<CrossRef<BatchTensor>>("end"),
                                      options.get<Size>("nstep"),
                                      options.get<Size>("dim"),
                                      options.get<Size>("batch_dim"))),
    UserTensor(options)
{
  auto bs = options.get<TensorShape>("batch_expand");
  if (bs.size() > 0)
    this->batch_expand(bs);
}
} // namespace neml2
