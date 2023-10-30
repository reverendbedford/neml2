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

#include "neml2/models/SR2Invariant.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(SR2Invariant);

OptionSet
SR2Invariant::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<LabeledAxisAccessor>("tensor");
  options.set<LabeledAxisAccessor>("invariant");
  options.set<std::string>("invariant_type");
  return options;
}

SR2Invariant::SR2Invariant(const OptionSet & options)
  : Model(options),
    tensor(declare_input_variable<SR2>(options.get<LabeledAxisAccessor>("tensor"))),
    invariant(declare_output_variable<Scalar>(options.get<LabeledAxisAccessor>("invariant"))),
    _type(options.get<std::string>("invariant_type"))
{
  setup();
}

void
SR2Invariant::set_value(const LabeledVector & in,
                        LabeledVector * out,
                        LabeledMatrix * dout_din,
                        LabeledTensor3D * d2out_din2) const
{
  const auto options = in.options();
  auto A = in.get<SR2>(tensor);

  if (_type == "I1")
  {
    if (out)
      out->set(A.tr(), invariant);
    if (dout_din)
      dout_din->set(SR2::identity(options), invariant, tensor);
    if (d2out_din2)
    {
      // zero
    }
  }
  else if (_type == "I2")
  {
    if (out)
      out->set((A.tr() * A.tr() - A.inner(A)) / 2.0, invariant);
    if (dout_din || d2out_din2)
    {
      auto I2 = SR2::identity(options);
      if (dout_din)
        dout_din->set(A.tr() * I2 - A, invariant, tensor);
      if (d2out_din2)
      {
        auto I2xI2 = SSR4::identity(options);
        auto I4sym = SSR4::identity_sym(options);
        d2out_din2->set(I2xI2 - I4sym, invariant, tensor, tensor);
      }
    }
  }
  else if (_type == "VONMISES")
  {
    auto S = A.dev();
    Scalar vm = std::sqrt(3.0 / 2.0) * S.norm(NEML2_EPS);

    if (out)
      out->set(vm, invariant);
    if (dout_din || d2out_din2)
    {
      auto dvm_dA = 3.0 / 2.0 * S / vm;
      if (dout_din)
        dout_din->set(dvm_dA, invariant, tensor);
      if (d2out_din2)
      {
        auto I = SSR4::identity_sym(options);
        auto J = SSR4::identity_dev(options);
        d2out_din2->set(
            3.0 / 2.0 * (I - 2.0 / 3.0 * dvm_dA.outer(dvm_dA)) * J / vm, invariant, tensor, tensor);
      }
    }
  }
  else
    throw NEMLException("Unsupported invariant type: " + _type);
}
} // namespace neml2
