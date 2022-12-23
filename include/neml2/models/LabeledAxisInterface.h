#pragma once

#include "neml2/tensors/LabeledAxis.h"

namespace neml2
{
/// Interface for an object that defines LabeledAxis
class LabeledAxisInterface
{
public:
  /// Setup the layouts of all the registered axes
  virtual void setup_layout();

protected:
  /// Declare an axis
  [[nodiscard]] LabeledAxis & declareAxis();

  /// Declare an item recursively on an axis
  template <typename T>
  LabeledAxisAccessor declareVariable(LabeledAxis & axis,
                                      const std::vector<std::string> & names) const
  {
    return declareVariable(axis, utils::storage_size(T::_base_sizes), names);
  }

  /// Declare an item (with known storage size) recursively on an axis
  LabeledAxisAccessor
  declareVariable(LabeledAxis & axis, TorchSize sz, const std::vector<std::string> & names) const
  {
    LabeledAxisAccessor accessor({names, sz});
    axis.add(accessor);
    return accessor;
  }

private:
  /// All the declared axes
  std::vector<std::shared_ptr<LabeledAxis>> _axes;
};
} // namespace neml2
