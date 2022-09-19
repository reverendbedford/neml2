#include "State.h"

using namespace torch::indexing;

State::State(const StateInfo & info, TorchSize nbatch) :
    StateBase(torch::empty(info.required_shape(nbatch))), 
    _info(info)
{
  setup_views();
}

State::State(const StateInfo & info, const torch::Tensor & tensor) :
    StateBase(tensor), _info(info)
{
  if (sizes() != _info.required_shape(sizes()[0]))
    throw std::runtime_error("Provided tensor does not have the correct shape "
                             "for the state described by the provided "
                             "StateInfo");
  setup_views();
}

State State::get_substate(std::string name)
{
  // No reshaping is needed, but we do need to grab the 
  // appropriate StateInfo
  return State(_info.substates().at(name), get_view(name));
}

const StateInfo & State::info() const
{
  return _info;
}

void State::setup_views()
{
  // Just add the items one by one as views into the flat array
  for (auto item : _info.item_locations()) {
    add_label(item.first, 
              {Slice(_info.item_offsets()[item.second],
                     _info.item_offsets()[item.second+1])});
  }

  // Can be done exactly the same as the actual objects
  for (auto item : _info.substate_locations()) {
    auto substate_info = _info.substates().at(item.first);
    add_label(item.first, 
              {Slice(_info.item_offsets()[item.second],
                     _info.item_offsets()[item.second + 
                     substate_info.nitems()])});
  }
}


