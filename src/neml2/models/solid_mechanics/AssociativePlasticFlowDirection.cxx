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

#include "neml2/models/solid_mechanics/AssociativePlasticFlowDirection.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
AssociativePlasticFlowDirection::AssociativePlasticFlowDirection(
    const std::string & name, const std::shared_ptr<YieldFunction> & f)
  : PlasticFlowDirection(name),
    yield_function(*f)
{
  register_model(f);
  setup();
}

void
AssociativePlasticFlowDirection::set_value(LabeledVector in,
                                           LabeledVector out,
                                           LabeledMatrix * dout_din) const
{
  // For associative flow, the flow direction is Np = df/dM
  TorchSize nbatch = in.batch_size();
  LabeledMatrix df_din(nbatch, yield_function.output(), yield_function.input());
  LabeledTensor<1, 3> d2f_din2(
      nbatch, yield_function.output(), yield_function.input(), yield_function.input());

  if (dout_din)
    std::tie(df_din, d2f_din2) = yield_function.dvalue_and_d2value(in);
  else
    df_din = yield_function.dvalue(in);

  auto df_dmandel = df_din.get<SymR2>(yield_function.yield_function, yield_function.mandel_stress);

  out.set(df_dmandel, plastic_flow_direction);

  if (dout_din)
  {
    // dNp/dM = d2f/dM2
    auto d2f_dmandel2 = d2f_din2.get<SymSymR4>(
        yield_function.yield_function, yield_function.mandel_stress, yield_function.mandel_stress);

    dout_din->set(d2f_dmandel2, plastic_flow_direction, yield_function.mandel_stress);
  }
}
} // namespace neml2
