#include "tensors/LabeledAxis.h"

namespace neml2
{
int LabeledAxis::level = 0;

LabeledAxis::LabeledAxis()
  : _offset(0)
{
}

LabeledAxis::LabeledAxis(const LabeledAxis & other)
  : _variables(other._variables),
    _layout(other._layout),
    _offset(0)
{
  // Deep copy the subaxis
  for (auto & [name, subaxis] : other._subaxes)
    _subaxes.emplace(name, std::make_shared<LabeledAxis>(*subaxis));
}

LabeledAxis &
LabeledAxis::add(const std::string & name, TorchSize sz)
{
  if (!has_variable(name))
    _variables.emplace(name, sz);
  return *this;
}

LabeledAxis &
LabeledAxis::merge(LabeledAxis & other)
{
  // First merge the variables
  _variables.insert(other._variables.begin(), other._variables.end());

  // Then merge the subaxes
  for (auto & [name, subaxis] : other._subaxes)
  {
    auto found = _subaxes.find(name);
    if (found == _subaxes.end())
      _subaxes.emplace(name, std::make_shared<LabeledAxis>(*subaxis));
    else
      found->second->merge(*subaxis);
  }

  return *this;
}

LabeledAxis &
LabeledAxis::prefix(const std::string & s, const std::string & delimiter)
{
  // Prefix all the variable names
  std::unordered_map<std::string, TorchSize> new_variables;
  for (const auto & [name, sz] : _variables)
    new_variables.emplace(s + delimiter + name, sz);
  _variables = new_variables;

  // Prefix all the subaxes
  std::unordered_map<std::string, std::shared_ptr<LabeledAxis>> new_subaxes;
  for (const auto & [name, subaxis] : _subaxes)
    new_subaxes.emplace(s + delimiter + name, subaxis);
  _subaxes = new_subaxes;

  return *this;
}

LabeledAxis &
LabeledAxis::suffix(const std::string & s, const std::string & delimiter)
{
  // Suffix all the variable names
  std::unordered_map<std::string, TorchSize> new_variables;
  for (const auto & [name, sz] : _variables)
    new_variables.emplace(name + delimiter + s, sz);
  _variables = new_variables;

  // Suffix all the subaxes
  std::unordered_map<std::string, std::shared_ptr<LabeledAxis>> new_subaxes;
  for (const auto & [name, subaxis] : _subaxes)
    new_subaxes.emplace(name + delimiter + s, subaxis);
  _subaxes = new_subaxes;

  return *this;
}

LabeledAxis &
LabeledAxis::rename(const std::string & original, const std::string & rename)
{
  // This could be a variable name
  auto var = _variables.find(original);
  if (var != _variables.end())
  {
    auto sz = var->second;
    _variables.erase(var);
    _variables.emplace(rename, sz);
    return *this;
  }

  // or a sub-axis name
  auto subaxis = _subaxes.find(original);
  if (subaxis != _subaxes.end())
  {
    auto axis = subaxis->second;
    _subaxes.erase(subaxis);
    _subaxes.emplace(rename, axis);
    return *this;
  }

  return *this;
}

LabeledAxis &
LabeledAxis::remove(const std::string & name)
{
  // This could be a variable name
  auto count = _variables.erase(name);
  if (count)
    return *this;

  // or a sub-axis name
  count += _subaxes.erase(name);

  // If nothing has been removed, we should probably notify the user.
  // Only do this in DEBUG, of course...
  if (count == 0)
  {
    std::cout << *this << std::endl;
    throw std::runtime_error("Nothing removed in LabeledAxis::remove, did you mispelled the name? "
                             "The LabeledAxis is print above if that helps.");
  }

  return *this;
}

LabeledAxis &
LabeledAxis::clear()
{
  _variables.clear();
  _subaxes.clear();
  _layout.clear();
  _offset = 0;

  return *this;
}

void
LabeledAxis::setup_layout()
{
  _offset = 0;
  _layout.clear();

  // First emplace all the variables
  std::map<std::string, TorchSize> sorted_variables(_variables.begin(), _variables.end());
  for (auto & [name, sz] : sorted_variables)
  {
    _layout.emplace(name, torch::indexing::Slice(_offset, _offset + sz));
    _offset += sz;
  }

  // Then subaxes
  std::map<std::string, std::shared_ptr<LabeledAxis>> sorted_subaxes(_subaxes.begin(),
                                                                     _subaxes.end());
  for (auto & [name, axis] : sorted_subaxes)
  {
    // Setup the sub-axis if necessary
    axis->setup_layout();
    _layout.emplace(name, torch::indexing::Slice(_offset, _offset + axis->storage_size()));
    _offset += axis->storage_size();
  }
}

TorchSize
LabeledAxis::storage_size(const std::string & name) const
{
  // This could be a variable name
  auto var = _variables.find(name);
  if (var != _variables.end())
    return var->second;

  // or a sub-axis name
  auto subaxis = _subaxes.find(name);
  if (subaxis != _subaxes.end())
    return subaxis->second->storage_size();

  // Only do this in DEBUG I guess
  throw std::runtime_error("In LabeledAxis::storage_size, no item matches given name " + name);
  return 0;
}

const TorchIndex &
LabeledAxis::indices(const std::string & name) const
{
  if (_layout.count(name) == 0)
    throw std::runtime_error("In LabeledAxis::indices, no item matches given name " + name);

  return _layout.at(name);
}

std::vector<std::string>
LabeledAxis::item_names() const
{
  std::vector<std::string> names;
  for (const auto & item : _layout)
    names.push_back(item.first);
  return names;
}

const LabeledAxis &
LabeledAxis::subaxis(const std::string & name) const
{
  if (_subaxes.count(name) == 0)
    throw std::runtime_error("In LabeledAxis::subaxis, no subaxis matches given name " + name);

  return *_subaxes.at(name);
}

LabeledAxis &
LabeledAxis::subaxis(const std::string & name)
{
  if (_subaxes.count(name) == 0)
    throw std::runtime_error("In LabeledAxis::subaxis, no subaxis matches given name " + name);

  return *_subaxes.at(name);
}

bool
LabeledAxis::equals(const LabeledAxis & other) const
{
  // They must have the same size
  if (_offset != other._offset)
    return false;

  // Comparing unordered maps is easy, two unordered_maps are equal if they have the same number of
  // elements and the elements in one container are a permutation of the elements in the other
  // container
  if (_variables != other._variables)
    return false;

  // For subaxes, it's a little bit tricky as we need to compare the dereferenced axes.
  for (auto & [name, axis] : _subaxes)
    if (other._subaxes.count(name) == 0)
      return false;
    else if (*other._subaxes.at(name) != *axis)
      return false;

  return true;
}

std::ostream &
operator<<(std::ostream & os, const LabeledAxis & info)
{
  os << utils::indentation(LabeledAxis::level);
  os << "LabeledAxis [" << info.storage_size() << "] {";
  if (info.nitem() == 0)
  {
    os << "}";
    return os;
  }

  LabeledAxis::level += 1;

  for (const auto & [name, sz] : info._variables)
    os << std::endl << utils::indentation(LabeledAxis::level) << name << ": [" << sz << "]";

  for (const auto & [name, axis] : info._subaxes)
  {
    os << std::endl;
    os << utils::indentation(LabeledAxis::level);
    os << name << ": " << std::endl;
    LabeledAxis::level += 1;
    os << *axis;
    LabeledAxis::level -= 1;
  }
  os << std::endl;
  os << utils::indentation(--LabeledAxis::level) << "}";
  return os;
}

void
LabeledAxis::to_dot(
    std::ostream & os, int & id, std::string axis_name, bool subgraph, bool node_handle) const
{
  // Preemble
  os << (subgraph ? "subgraph " : "graph ");
  os << "cluster_" << id++ << " ";
  os << "{\n";
  os << "label = \"" << axis_name << "\"\n";
  os << "bgcolor = lightgrey\n";

  // The axis should have an invisible node so that I can draw arrows
  if (node_handle)
    os << "\"" << axis_name << "\" [label = \"\", style = invis]\n";

  // Write all the variables
  for (const auto & [name, sz] : _variables)
  {
    os << "\"" << axis_name + " " + name << "\" ";
    os << "[style = filled, color = white, shape = Square, ";
    os << "label = \"" << name + " [" << sz << "]\"]\n";
  }

  // Write all the subaxes
  for (const auto & [name, subaxis] : _subaxes)
    subaxis->to_dot(os, id, axis_name + " " + name, true);

  os << "}\n";
}

bool
operator==(const LabeledAxis & a, const LabeledAxis & b)
{
  return a.equals(b);
}

bool
operator!=(const LabeledAxis & a, const LabeledAxis & b)
{
  return !a.equals(b);
}
} // namespace neml2
