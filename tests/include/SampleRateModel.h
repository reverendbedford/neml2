#pragma once

#include "models/Model.h"
#include "models/ADModel.h"

template <bool is_ad>
using SampleRateModelBase = std::conditional_t<is_ad, ADModel, Model>;

// A dummy rate model for testing purposes
template <bool is_ad>
class SampleRateModelTempl : public SampleRateModelBase<is_ad>
{
public:
  SampleRateModelTempl(const std::string & name);

  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};

typedef SampleRateModelTempl<true> ADSampleRateModel;
typedef SampleRateModelTempl<false> SampleRateModel;
