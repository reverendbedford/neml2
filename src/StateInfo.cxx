#include "StateInfo.h"

StateInfo::StateInfo()
  : _item_offsets({0})
{
}

void
StateInfo::add_substate(std::string name, const StateInfo & substate)
{
  // Store the location of the start of these items
  _substate_locations.insert({name, nitems()});

  // Store the object itself to help with reconstructing it later
  _substates.insert({name, substate});

  // Loop through the substate items and add them to this object
  for (auto it : substate.item_locations())
  {
    // So I stop confusing myself
    std::string item_name = it.first;
    size_t loc = it.second;
    TorchSize item_size = substate.item_offsets()[loc + 1] - substate.item_offsets()[loc];
    _item_locations.insert({item_name, nitems()});
    _item_offsets.push_back(size_storage() + item_size);
  }
}

size_t
StateInfo::nitems() const
{
  return _item_locations.size();
}

TorchSize
StateInfo::size_storage() const
{
  return _item_offsets.back();
}

size_t
StateInfo::nsubstates() const
{
  return _substates.size();
}

TorchShape
StateInfo::required_shape(TorchSize nbatch) const
{
  return TorchShape({nbatch, size_storage()});
}

TorchSize
StateInfo::base_storage(std::string item) const
{
  size_t loc = _item_locations.at(item);
  return _item_offsets[loc + 1] - _item_offsets[loc];
}
