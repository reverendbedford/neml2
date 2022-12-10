#pragma once

#include "models/SecDerivModel.h"

namespace neml2
{
/// Parent class for all yield functions
class YieldFunction : public SecDerivModel
{
public:
  /// Calculate yield function knowing the corresponding hardening model
  YieldFunction(const std::string & name);

protected:
  const LabeledAxisAccessor _mandel_idx;
  const LabeledAxisAccessor _f_idx;
};
} // namespace neml2
