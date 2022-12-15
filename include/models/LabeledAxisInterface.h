#pragma once

#include "tensors/LabeledAxis.h"

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
  template <typename T,
            typename... S,
            typename = std::enable_if_t<are_all_convertible<std::string, S...>::value>>
  LabeledAxisAccessor declareVariable(LabeledAxis & axis, S &&... name) const
  {
    return declareVariable(axis, utils::storage_size(T::_base_sizes), std::forward<S>(name)...);
  }

  /// Declare an item (with known storage size) recursively on an axis
  template <typename... S>
  LabeledAxisAccessor declareVariable(LabeledAxis & axis, TorchSize sz, S &&... name) const
  {
    std::vector<std::string> names({name...});
    declareVariableHelper(axis, sz, names.begin(), names.end());
    return LabeledAxisAccessor({names, sz});
  }

private:
  /// Helper method to declare a variable recursively
  void declareVariableHelper(LabeledAxis & axis,
                             TorchSize sz,
                             const std::vector<std::string>::const_iterator & cur,
                             const std::vector<std::string>::const_iterator & end) const
  {
    if (cur == end - 1)
      axis.add(*cur, sz);
    else
    {
      axis.add<LabeledAxis>(*cur);
      declareVariableHelper(axis.subaxis(*cur), sz, cur + 1, end);
    }
  }

  /// All the declared axes
  std::vector<std::shared_ptr<LabeledAxis>> _axes;
};
} // namespace neml2
