#pragma once

#include "models/Model.h"

// A dummy rate model for testing purposes
class SampleRateModel : public Model
{
public:
  SampleRateModel(const std::string & name);

  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;
};
