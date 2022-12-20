#pragma once

#include "models/SecDerivModel.h"

namespace neml2
{
/// Parent class for all stress measures
class StressMeasure : public SecDerivModel
{
public:
  StressMeasure(const std::string & name);

  const LabeledAxisAccessor overstress;
  const LabeledAxisAccessor stress_measure;
};
} // namespace neml2
