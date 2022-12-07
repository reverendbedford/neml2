#pragma once

#include "models/SecDerivModel.h"
#include "models/ADSecDerivModel.h"

template <bool is_ad>
using SampleSecDerivModelBase = std::conditional_t<is_ad, ADSecDerivModel, SecDerivModel>;

// A dummy rate model for testing purposes
template <bool is_ad>
class SampleSecDerivModelTempl : public SampleSecDerivModelBase<is_ad>
{
public:
  SampleSecDerivModelTempl(const std::string & name);

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  virtual void set_dvalue(LabeledVector in,
                          LabeledMatrix dout_din,
                          LabeledTensor<1, 3> * d2out_din2 = nullptr) const;
};

typedef SampleSecDerivModelTempl<true> ADSampleSecDerivModel;
typedef SampleSecDerivModelTempl<false> SampleSecDerivModel;
