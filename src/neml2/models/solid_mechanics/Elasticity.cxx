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

#include "neml2/models/solid_mechanics/Elasticity.h"

namespace neml2
{
ParameterSet
Elasticity::expected_params()
{
  ParameterSet params = Model::expected_params();
  params.set<LabeledAxisAccessor>("strain") = {{"state", "internal", "Ee"}};
  params.set<LabeledAxisAccessor>("stress") = {{"state", "S"}};
  params.set<bool>("compliance") = false;
  params.set<bool>("rate_form") = false;
  return params;
}

Elasticity::Elasticity(const ParameterSet & params)
  : Model(params),
    _compliance(params.get<bool>("compliance")),
    _rate_form(params.get<bool>("rate_form")),
    _strain(params.get<LabeledAxisAccessor>("strain").with_suffix(_rate_form ? "_rate" : "")),
    _stress(params.get<LabeledAxisAccessor>("stress").with_suffix(_rate_form ? "_rate" : "")),
    from_var(declare_input_variable<SymR2>(_compliance ? _stress : _strain)),
    to_var(declare_output_variable<SymR2>(_compliance ? _strain : _stress))
{
  setup();
}
} // namespace neml2
