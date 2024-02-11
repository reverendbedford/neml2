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

#include "neml2/models/RotationMatrix.h"

namespace neml2
{
register_NEML2_object(RotationMatrix);

OptionSet
RotationMatrix::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<VariableName>("from");
  options.set<VariableName>("to");
  return options;
}

RotationMatrix::RotationMatrix(const OptionSet & options)
  : Model(options),
    _from(declare_input_variable<Rot>("from")),
    _to(declare_output_variable<R2>("to"))
{
}

void
RotationMatrix::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert(!d2out_din2, "Second derivatives not implemented");

  if (out)
    _to = Rot(_from).euler_rodrigues();

  if (dout_din)
    _to.d(_from) = Rot(_from).deuler_rodrigues();
}
} // namespace neml2
