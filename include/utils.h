#pragma once

#include "types.h"

namespace utils
{
constexpr double sqrt2 = 1.4142135623730951;

inline constexpr double
mandelFactor(TorchSize i)
{
  return i < 3 ? 1.0 : sqrt2;
}
}
