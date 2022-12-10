#include "models/ImplicitModel.h"

namespace neml2
{
ImplicitModel::Stage ImplicitModel::stage = UPDATING;

BatchTensor<1>
ImplicitModel::initial_guess(LabeledVector in) const
{
  // By default use the old state as the initial guess
  return in("old_state");
}

void
ImplicitModel::cache_input(LabeledVector in)
{
  _cached_in = in.clone();
}
} // namespace neml2
