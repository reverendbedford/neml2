#pragma once

#include "StateProvider.h"
#include "StateFunction.h"

class HardeningMap : public SingleStateFunction, public StateProvider
{
public:
  /// A map between the state and the hardening variable names
  //  The default implementation just appends "conjugate" to the
  //  internal variable name, but overriding in subclasses lets you
  //  assign nice names instead.
  virtual std::string conjugate_name(std::string stress_var) const;
};
