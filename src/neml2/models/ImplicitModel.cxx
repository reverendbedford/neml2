#include "neml2/models/ImplicitModel.h"

namespace neml2
{
ImplicitModel::Stage ImplicitModel::stage = UPDATING;

BatchTensor<1>
ImplicitModel::initial_guess(LabeledVector in) const
{
  LabeledVector guess(in.batch_size(), input().subaxis("old_state"));
  guess.fill(in.slice("old_state"));
  return guess.tensor();
}

void
ImplicitModel::cache_input(LabeledVector in)
{
  _cached_in = in.clone();
}
} // namespace neml2
