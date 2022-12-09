#include "models/LabeledAxisInterface.h"

namespace neml2
{
LabeledAxis &
LabeledAxisInterface::declareAxis()
{
  _axes.push_back(std::make_shared<LabeledAxis>());
  return *_axes.back();
}

void
LabeledAxisInterface::setup_layout()
{
  for (auto & axis : _axes)
    axis->setup_layout();
}
} // namespace neml2
