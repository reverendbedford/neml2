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


#include "neml2/models/solid_mechanics/TotalStrain.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
template <bool rate>
TotalStrainTempl<rate>::TotalStrainTempl(const std::string & name)
  : Model(name),
    elastic_strain(
        declareInputVariable<SymR2>({"state", rate ? "elastic_strain_rate" : "elastic_strain"})),
    plastic_strain(
        declareInputVariable<SymR2>({"state", rate ? "plastic_strain_rate" : "plastic_strain"})),
    total_strain(
        declareOutputVariable<SymR2>({"state", rate ? "total_strain_rate" : "total_strain"}))
{
  this->setup();
}

template <bool rate>
void
TotalStrainTempl<rate>::set_value(LabeledVector in,
                                  LabeledVector out,
                                  LabeledMatrix * dout_din) const
{
  // total strain = elastic strain + plastic strain
  out.set(in.get<SymR2>(elastic_strain) + in.get<SymR2>(plastic_strain), total_strain);

  if (dout_din)
  {
    auto I = SymR2::identity_map().batch_expand(in.batch_size());
    dout_din->set(I, total_strain, elastic_strain);
    dout_din->set(I, total_strain, plastic_strain);
  }
}

template class TotalStrainTempl<true>;
template class TotalStrainTempl<false>;
} // namespace neml2
