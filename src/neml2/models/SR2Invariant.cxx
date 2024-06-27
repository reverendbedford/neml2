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
  options.doc() = "Calculate the invariant of a symmetric second order tensor (of type SR2).";

  options.set<VariableName>("tensor");
  options.set("tensor").doc() = "SR2 which is used to calculate the invariant of";

  options.set<VariableName>("invariant");
  options.set("invariant").doc() = "Invariant";

  options.set<std::string>("invariant_type");
  options.set("invariant_type").doc() = "Type of invariant. Options are I1, I2, and VONMISES.";

  return options;
}

SR2Invariant::SR2Invariant(const OptionSet & options)
  : Model(options),
    _type(options.get<std::string>("invariant_type")),
    _A(declare_input_variable<SR2>("tensor")),
    _invariant(declare_output_variable<Scalar>("invariant"))
{
}

void
SR2Invariant::set_value(bool out, bool dout_din, bool d2out_din2)
{
  auto A = SR2(_A);

  if (_type == "I1")
  {
    if (out)
      _invariant = A.tr();
    if (dout_din)
      _invariant.d(_A) = SR2::identity(options());
    if (d2out_din2)
    {
      // zero
    }
  }
  else if (_type == "I2")
  {
    if (out)
      _invariant = (A.tr() * A.tr() - A.inner(A)) / 2.0;
    if (dout_din || d2out_din2)
    {
      auto I2 = SR2::identity(options());
      if (dout_din)
        _invariant.d(_A) = A.tr() * I2 - A;
      if (d2out_din2)
      {
        auto I2xI2 = SSR4::identity(options());
        auto I4sym = SSR4::identity_sym(options());
        _invariant.d(_A, _A) = I2xI2 - I4sym;
      }
    }
  }
  else if (_type == "VONMISES")
  {
    auto S = A.dev();
    Scalar vm = std::sqrt(3.0 / 2.0) * S.norm(machine_precision());

    if (out)
      _invariant = vm;
    if (dout_din || d2out_din2)
    {
      auto dvm_dA = 3.0 / 2.0 * S / vm;
      if (dout_din)
        _invariant.d(_A) = dvm_dA;
      if (d2out_din2)
      {
        auto I = SSR4::identity_sym(options());
        auto J = SSR4::identity_dev(options());
        _invariant.d(_A, _A) = 3.0 / 2.0 * (I - 2.0 / 3.0 * dvm_dA.outer(dvm_dA)) * J / vm;
      }
    }
  }
  else if (_type == "EFFECTIVE_STRAIN")
  {
    Scalar r = std::sqrt(2.0 / 3.0) * A.norm(machine_precision());

    if (out)
      _invariant = r;

    if (dout_din || d2out_din2)
    {
      auto d = 2.0 / 3.0 * A / r;

      if (dout_din)
        _invariant.d(_A) = 2.0 / 3.0 * A / r;

      if (d2out_din2)
        _invariant.d(_A, _A) =
            2.0 / 3.0 * (SSR4::identity_sym(options()) - 3.0 / 2.0 * d.outer(d)) / r;
    }
  }
  else
    throw NEMLException("Unsupported invariant type: " + _type);
}
} // namespace neml2
