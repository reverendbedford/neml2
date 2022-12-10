#pragma once

#include "tensors/LabeledAxis.h"

namespace neml2
{
/// Interface for an object that defines LabeledAxis
class LabeledAxisInterface
{
public:
  /// Declare an axis
  [[nodiscard]] LabeledAxis & declareAxis();

  /// Declare an item recursively on an axis
  template <typename T, typename... S>
  [[nodiscard]] LabeledAxisAccessor declareVariable(LabeledAxis & axis, S &&... name) const
  {
    std::vector<std::string> names({name...});
    declareVariableHelper<T>(axis, names.begin(), names.end());
    return LabeledAxisAccessor({names});
  }

  /// Setup the layouts of all the registered axes
  virtual void setup_layout();

private:
  /// Helper method to declare a variable recursively
  template <typename T>
  void declareVariableHelper(LabeledAxis & axis,
                             const std::vector<std::string>::const_iterator & cur,
                             const std::vector<std::string>::const_iterator & end) const
  {
    if (cur == end - 1)
      axis.add<T>(*cur);
    else
    {
      axis.add<LabeledAxis>(*cur);
      declareVariableHelper<T>(axis.subaxis(*cur), cur + 1, end);
    }
  }

  std::vector<std::shared_ptr<LabeledAxis>> _axes;
};
} // namespace neml2
