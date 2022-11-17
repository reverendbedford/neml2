#pragma once

#include "models/solid_mechanics/HardeningMap.h"

class IsotropicHardeningMap : public HardeningMap
{
public:
  /// State (just equivalent_plastic_strain)
  virtual StateInfo state() const;
  /// Initial state
  virtual void initial_state(State & input) const;

  /// The output is just "isotropic_hardening"
  virtual StateInfo output() const;
  /// Give the conjugate variable the nice name "equivalent_plastic_strain"
  virtual std::string conjugate_name(std::string stress_var) const;
};
