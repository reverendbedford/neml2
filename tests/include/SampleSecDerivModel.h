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

#include "neml2/models/SecDerivModel.h"
#include "neml2/models/ADSecDerivModel.h"

template <bool is_ad>
using SampleSecDerivModelBase =
    std::conditional_t<is_ad, neml2::ADSecDerivModel, neml2::SecDerivModel>;

template <bool is_ad>
class SampleSecDerivModelTempl : public SampleSecDerivModelBase<is_ad>
{
public:
  SampleSecDerivModelTempl(const std::string & name);

protected:
  virtual void set_value(neml2::LabeledVector in,
                         neml2::LabeledVector out,
                         neml2::LabeledMatrix * dout_din = nullptr) const;

  virtual void set_dvalue(neml2::LabeledVector in,
                          neml2::LabeledMatrix dout_din,
                          neml2::LabeledTensor<1, 3> * d2out_din2 = nullptr) const;
};

typedef SampleSecDerivModelTempl<true> ADSampleSecDerivModel;
typedef SampleSecDerivModelTempl<false> SampleSecDerivModel;
