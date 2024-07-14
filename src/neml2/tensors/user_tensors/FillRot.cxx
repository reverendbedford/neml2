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

#include "neml2/tensors/user_tensors/FillRot.h"

namespace neml2
{
register_NEML2_object(FillRot);

OptionSet
FillRot::expected_options()
{
  OptionSet options = UserTensorBase::expected_options();
  options.doc() = "Construct a Rot from a vector of Scalars.";

  options.set<std::vector<CrossRef<Scalar>>>("values");
  options.set("values").doc() = "Scalars used to fill the Rot";

  options.set<std::string>("method") = "modified";
  options.set("method").doc() = "Fill method, options are 'modified' and 'standard'.";

  return options;
}

FillRot::FillRot(const OptionSet & options)
  : Rot(fill(options.get<std::vector<CrossRef<Scalar>>>("values"),
             options.get<std::string>("method"))),
    UserTensorBase(options)
{
}

Rot
FillRot::fill(const std::vector<CrossRef<Scalar>> & values, const std::string & method) const
{
  if (method == "modified")
  {
    if (values.size() == 3)
      return Rot::fill(values[0], values[1], values[2]);
    else
      neml_assert(
          false, "Number of values must be 3, but ", values.size(), " values are provided.");
  }
  else if (method == "standard")
  {
    if (values.size() == 3)
    {
      auto ns = values[0] * values[0] + values[1] * values[1] + values[2] * values[2];
      auto f = Scalar(torch::sqrt(torch::Tensor(ns) + torch::tensor(1.0, ns.dtype())) +
                      torch::tensor(1.0, ns.dtype()));
      return Rot::fill(values[0] / f, values[1] / f, values[2] / f);
    }
    else
      neml_assert(
          false, "Number of values must be 3, but ", values.size(), " values are provided.");
  }
  else
    throw NEMLException("Unknown Rot fill type " + method);
  return Rot();
}
} // namespace neml2
