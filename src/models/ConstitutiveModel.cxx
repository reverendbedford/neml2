#include "models/ConstitutiveModel.h"
#include "state/StateInfo.h"

State
ConstitutiveModel::state_update(const State & forces_np1,
                                const State & state_n,
                                const State & forces_n)
{
  // Can infer the batch size from the others
  if ((forces_np1.batch_size() != state_n.batch_size()) ||
      (state_n.batch_size() != forces_n.batch_size()))
    throw std::runtime_error("Input batch sizes do not agree");

  return value({forces_np1, state_n, forces_n});
}

StateDerivativeOutput
ConstitutiveModel::linearized_state_update(const State & forces_np1,
                                           const State & state_n,
                                           const State & forces_n)
{
  // Can infer the batch size from the others
  if ((forces_np1.batch_size() != state_n.batch_size()) ||
      (state_n.batch_size() != forces_n.batch_size()))
    throw std::runtime_error("Input batch sizes do not agree");

  return dvalue({forces_np1, state_n, forces_n});
}

StateInfo
ConstitutiveModel::output() const
{
  return state();
}
