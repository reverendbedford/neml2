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

  /// Setup the layouts of all the registered axes
  virtual void setup_layout();

private:
  std::vector<std::shared_ptr<LabeledAxis>> _axes;
};
} // namespace neml2
