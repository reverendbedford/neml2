#pragma once

#include "models/SecDerivModel.h"
#include "models/ADSecDerivModel.h"

template <bool is_ad>
using SampleSecDerivModelBase =
    std::conditional_t<is_ad, neml2::ADSecDerivModel, neml2::SecDerivModel>;

// A dummy rate model for testing purposes
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
