#pragma once

#include "StateFunction.h"

/// Defines the consistency parameter
class InelasticFlowRate : public SingleStateFunction
{
public:
  virtual StateInfo output() const;
};
