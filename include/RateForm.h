#pragma once

#include "ConstitutiveModel.h"

/// Model defined in terms of the *rate* of state evolution
//    value maps state, forces, and forces_rate -> state_rate
class RateForm : public ConstitutiveModel
{
public:
  virtual StateInfo output() const;
};
