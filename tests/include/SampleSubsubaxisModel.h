#pragma once

#include "models/Model.h"

// A dummy model containing a sub-sub-axis for testing purposes
class SampleSubsubaxisModel : public neml2::Model
{
public:
  SampleSubsubaxisModel(const std::string & name);

  const neml2::LabeledAxisAccessor foo;
  const neml2::LabeledAxisAccessor bar;
  const neml2::LabeledAxisAccessor baz;

protected:
  virtual void set_value(neml2::LabeledVector in,
                         neml2::LabeledVector out,
                         neml2::LabeledMatrix * dout_din = nullptr) const;
};
