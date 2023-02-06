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

#include "neml2/models/solid_mechanics/ElasticStrain.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
register_NEML2_object(ElasticStrain);
register_NEML2_object(ElasticStrainRate);

template <bool rate>
ParameterSet
ElasticStrainTempl<rate>::expected_params()
{
  ParameterSet params = Model::expected_params();
  return params;
}

template <bool rate>
ElasticStrainTempl<rate>::ElasticStrainTempl(const ParameterSet & params)
  : Model(params),
    total_strain(
        declareInputVariable<SymR2>({"forces", rate ? "total_strain_rate" : "total_strain"})),
    plastic_strain(
        declareInputVariable<SymR2>({"state", rate ? "plastic_strain_rate" : "plastic_strain"})),
    elastic_strain(
        declareOutputVariable<SymR2>({"state", rate ? "elastic_strain_rate" : "elastic_strain"}))
{
  setup();
}

template <bool rate>
void
ElasticStrainTempl<rate>::set_value(LabeledVector in,
                                    LabeledVector out,
                                    LabeledMatrix * dout_din) const
{
  // Simple additive decomposition:
  // elastic strain = total strain - plastic strain
  out.set(in.get<SymR2>(total_strain) - in.get<SymR2>(plastic_strain), elastic_strain);

  if (dout_din)
  {
    auto I = SymR2::identity_map().batch_expand(in.batch_size());
    dout_din->set(I, elastic_strain, total_strain);
    dout_din->set(-I, elastic_strain, plastic_strain);
  }
}

template class ElasticStrainTempl<true>;
template class ElasticStrainTempl<false>;
} // namespace neml2
