#include "State.h"

using namespace torch::indexing;

State::State(const StateInfo & info, TorchSize nbatch) :
    StandardBatchedLabeledTensor(torch::empty(info.required_shape(nbatch))), 
    _info(info)
{
  setup_views();
}

State::State(const StateInfo & info, const torch::Tensor & tensor) :
    StandardBatchedLabeledTensor(tensor), _info(info)
{
  if (sizes() != _info.required_shape(sizes()[0]))
    throw std::runtime_error("Provided tensor does not have the correct shape "
                             "for the state described by the provided "
                             "StateInfo");
  setup_views();
}

TorchSize State::batch_size() const
{
  return sizes()[0];
}

void State::setup_views()
{
  for (auto item : _info.item_locations()) {
    add_label(item.first, 
              {Slice(_info.item_offsets()[item.second],
                     _info.item_offsets()[item.second+1])});
  }

  // For now reuse the string -> Slice mechanism by just adding 
  // substate_prefix to the start of each substate slice name
  // TODO: finish all of this
}
