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

#pragma once

#include "neml2/models/Model.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
enum ElasticityType
{
  STIFFNESS,
  COMPLIANCE
};

template <bool rate, ElasticityType etype>
class LinearElasticity : public Model
{
public:
  LinearElasticity(const std::string & name, SymSymR4 T);

  static constexpr std::string in_name();
  static constexpr std::string out_name();

  const LabeledAxisAccessor from;
  const LabeledAxisAccessor to;

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  /**
  The fourth order transformation tensor. When `etype == STIFFNESS`, this is the stiffness tensor;
  when `etype == COMPLIANCE`, this is the compliance tensor.
  */
  SymSymR4 _T;
};

typedef LinearElasticity<false, ElasticityType::STIFFNESS> CauchyStressFromElasticStrain;
typedef LinearElasticity<false, ElasticityType::COMPLIANCE> ElasticStrainFromCauchyStress;
typedef LinearElasticity<true, ElasticityType::STIFFNESS> CauchyStressRateFromElasticStrainRate;
typedef LinearElasticity<true, ElasticityType::COMPLIANCE> ElasticStrainRateFromCauchyStressRate;
} // namespace neml2
