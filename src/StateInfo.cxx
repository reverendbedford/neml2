#include "StateInfo.h"

StateInfo::StateInfo()
  : _item_offsets({0})
{
}

StateInfo::StateInfo(std::map<std::string, size_t> locations, std::vector<TorchSize> offsets)
  : _item_locations(locations),
    _item_offsets(offsets)
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

StateInfo &
StateInfo::add_prefix(std::string prefix)
{
  std::map<std::string, size_t> new_locations;
  for (auto it : _item_locations)
    new_locations.insert({prefix + it.first, it.second});
  _item_locations = new_locations;

  std::map<std::string, size_t> new_substate_locations;
  for (auto it : _substate_locations)
    new_substate_locations.insert({prefix + it.first, it.second});
  _substate_locations = new_substate_locations;

  std::map<std::string, StateInfo> new_substates;
  for (auto it : _substates)
    new_substates.insert({prefix + it.first, it.second});
  _substates = new_substates;

  return *this;
}

StateInfo &
StateInfo::add_suffix(std::string suffix)
{
  std::map<std::string, size_t> new_locations;
  for (auto it : _item_locations)
    new_locations.insert({it.first + suffix, it.second});
  _item_locations = new_locations;

  std::map<std::string, size_t> new_substate_locations;
  for (auto it : _substate_locations)
    new_substate_locations.insert({it.first + suffix, it.second});
  _substate_locations = new_substate_locations;

  std::map<std::string, StateInfo> new_substates;
  for (auto it : _substates)
    new_substates.insert({it.first + suffix, it.second});
  _substates = new_substates;

  return *this;
}

std::vector<std::string>
StateInfo::items() const
{
  std::vector<std::string> names;
  for (auto it : _item_locations)
    names.push_back(it.first);
  return names;
}

std::vector<std::string>
StateInfo::substate_names() const
{
  std::vector<std::string> names;
  for (auto it : _substate_locations)
    names.push_back(it.first);
  return names;
}

bool
StateInfo::equals(const StateInfo & other) const
{
  return (nitems() == other.nitems() && (size_storage() == other.size_storage()) &&
          (item_locations() == other.item_locations()) &&
          (item_offsets() == other.item_offsets()) && (nsubstates() == other.nsubstates()) &&
          (substate_locations() == other.substate_locations()) &&
          (substates() == other.substates()));
}

StateInfo &
StateInfo::rename(std::string original, std::string rename)
{
  std::map<std::string, size_t> new_locations;
  for (auto it : _item_locations)
  {
    if (it.first == original)
      new_locations.insert({rename, it.second});
    else
      new_locations.insert({it.first, it.second});
  }
  _item_locations = new_locations;

  return *this;
}

StateInfo
StateInfo::remove(std::string name) const
{
  // There is a better way to do this...
  StateInfo info;
  for (auto item : items())
  {
    if (name == item)
      continue;
    info.add(item, base_storage(item));
  }
  return info;
}

void
StateInfo::add(std::string name, size_t sz)
{
  _item_locations.insert({name, nitems()});
  _item_offsets.push_back(size_storage() + sz);
}

bool
operator==(const StateInfo & a, const StateInfo & b)
{
  return a.equals(b);
}

bool
operator!=(const StateInfo & a, const StateInfo & b)
{
  return !a.equals(b);
}
