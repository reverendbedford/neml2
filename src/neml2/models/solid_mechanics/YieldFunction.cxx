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

#include "neml2/models/solid_mechanics/YieldFunction.h"

namespace neml2
{
register_NEML2_object(PerfectlyPlasticYieldFunction);
register_NEML2_object(IsotropicHardeningYieldFunction);
register_NEML2_object(KinematicHardeningYieldFunction);
register_NEML2_object(IsotropicAndKinematicHardeningYieldFunction);

template <bool isoharden, bool kinharden>
ParameterSet
YieldFunction<isoharden, kinharden>::expected_params()
{
  ParameterSet params = YieldFunctionBase::expected_params();
  params.set<bool>("with_isotropic_hardening") = isoharden;
  params.set<bool>("with_kinematic_hardening") = kinharden;
  return params;
}

template class YieldFunction<false, false>;
template class YieldFunction<true, false>;
template class YieldFunction<false, true>;
template class YieldFunction<true, true>;
} // namespace neml2
