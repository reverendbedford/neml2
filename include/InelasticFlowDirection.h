#pragma once

#include "StateFunction.h"

/// Defines the direction of plastic flow
class InelasticFlowDirection : public SingleStateFunction
{
public:
  /// Object output
  virtual StateInfo output() const;
};
