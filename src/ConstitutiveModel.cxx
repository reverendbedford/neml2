#include "ConstitutiveModel.h"

State
ConstitutiveModel::update_state(const State & forces_np1, 
                                const State & state_n, 
                                const State & forces_n)
{
  // Can infer the batch size from the others
  if (
      (forces_np1.batch_size() != state_n.batch_size()) ||
      (state_n.batch_size() != forces_n.batch_size())
     )
    throw std::runtime_error("Input batch sizes do not agree");

  State state_np1(state(), forces_np1.batch_size());

  update(state_np1, forces_np1, state_n, forces_n);

  return state_np1;
}
