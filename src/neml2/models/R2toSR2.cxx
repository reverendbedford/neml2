// Copyright 2024, UChicago Argonne, LLC
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

#include "neml2/models/R2toSR2.h"

namespace neml2
{
register_NEML2_object(R2toSR2);

OptionSet
R2toSR2::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Extract the symmetric part of a R2 tensor";

  options.set_input("input");
  options.set("input").doc() = "Rank two tensor to split";

  options.set_output("output");
  options.set("output").doc() = "Output symmetric rank two tensor";

  return options;
}

R2toSR2::R2toSR2(const OptionSet & options)
  : Model(options),
    _input(declare_input_variable<R2>("input")),
    _output(declare_output_variable<SR2>("output"))
{
}

void
R2toSR2::set_value(bool out, bool dout_din, bool d2out_din2)
{
  auto A = R2(_input);

  if (out)
    _output = SR2(A);

  (void)dout_din;
  (void)d2out_din2;
}
} // namespace neml2
