#pragma once

#include "neml2/models/Model.h"
#include "neml2/models/ADModel.h"

template <bool is_ad>
using SampleRateModelBase = std::conditional_t<is_ad, neml2::ADModel, neml2::Model>;

// A dummy rate model for testing purposes
template <bool is_ad>
class SampleRateModelTempl : public SampleRateModelBase<is_ad>
{
public:
  SampleRateModelTempl(const std::string & name);

protected:
  virtual void set_value(neml2::LabeledVector in,
                         neml2::LabeledVector out,
                         neml2::LabeledMatrix * dout_din = nullptr) const;
};

typedef SampleRateModelTempl<true> ADSampleRateModel;
typedef SampleRateModelTempl<false> SampleRateModel;
