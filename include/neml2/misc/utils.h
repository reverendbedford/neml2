#pragma once

#include "neml2/misc/types.h"

namespace neml2
{
namespace utils
{
constexpr double sqrt2 = 1.4142135623730951;

inline constexpr double
mandelFactor(TorchSize i)
{
  return i < 3 ? 1.0 : sqrt2;
}

/// Helper to get the total storage required from a TorchShape
inline TorchSize
storage_size(const TorchShape & shape)
{
  TorchSize sz = 1;
  return std::accumulate(shape.begin(), shape.end(), sz, std::multiplies<TorchSize>());
}

/// Generically useful helper function that merges multiple TorchShapes
template <typename... TorchShapeRef>
inline TorchShape
add_shapes(TorchShapeRef... shapes)
{
  TorchShape net;
  (net.insert(net.end(), shapes.begin(), shapes.end()), ...);
  return net;
}

std::string indentation(int level, int indent = 2);

template <typename T>
std::string
stringify(const T & t)
{
  std::ostringstream os;
  os << t;
  return os.str();
}
} // namespace utils
} // namespace neml2
