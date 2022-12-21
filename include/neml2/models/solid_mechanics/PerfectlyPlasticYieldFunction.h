#pragma once

#include "neml2/models/solid_mechanics/YieldFunction.h"

namespace neml2
{

/// Yield function with no hardening
class PerfectlyPlasticYieldFunction : public YieldFunction
{
public:
  PerfectlyPlasticYieldFunction(const std::string & name,
                                const std::shared_ptr<StressMeasure> & sm,
                                Scalar s0);
};

} // namespace neml2
