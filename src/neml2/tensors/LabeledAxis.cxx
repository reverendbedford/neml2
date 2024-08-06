// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "neml2/tensors/LabeledAxis.h"

namespace neml2
{
LabeledAxis::LabeledAxis()
  : _offset(0),
    _has_state(false),
    _has_old_state(false),
    _has_forces(false),
    _has_old_forces(false),
    _has_residual(false),
    _has_parameters(false)
{
}

LabeledAxis::LabeledAxis(const LabeledAxis & other)
  : _variables(other._variables),
    _subaxes(other._subaxes),
    _layout(other._layout),
    _offset(other._offset),
    _has_state(other._has_state),
    _has_old_state(other._has_old_state),
    _has_forces(other._has_forces),
    _has_old_forces(other._has_old_forces),
    _has_residual(other._has_residual),
    _has_parameters(other._has_parameters)
{
}

LabeledAxis &
LabeledAxis::add(const LabeledAxisAccessor & accessor, Size sz)
{
  if (!accessor.empty())
    add(*this, sz, accessor.begin(), accessor.end());
  return *this;
}

void
LabeledAxis::add(LabeledAxis & axis,
                 Size sz,
                 const LabeledAxisAccessor::const_iterator & cur,
                 const LabeledAxisAccessor::const_iterator & end) const
{
  if (cur == end - 1)
  {
    if (!axis.has_variable(*cur))
      axis._variables.emplace(*cur, sz);
  }
  else
  {
    axis.add<LabeledAxis>(*cur);
    add(axis.subaxis(*cur), sz, cur + 1, end);
  }
}

void
LabeledAxis::clear()
{
  _variables.clear();
  _subaxes.clear();
  _layout.clear();
  _offset = 0;
}

void
LabeledAxis::setup_layout()
{
  _offset = 0;
  _layout.clear();

  // First emplace all the variables
  for (auto & [name, sz] : _variables)
  {
    std::pair<Size, Size> range = {_offset, _offset + sz};
    _layout.emplace(name, range);
    _offset += sz;
  }

  // Then subaxes
  for (auto & [name, axis] : _subaxes)
  {
    // Setup the sub-axis if necessary
    axis->setup_layout();
    std::pair<Size, Size> range = {_offset, _offset + axis->storage_size()};
    _layout.emplace(name, range);
    _offset += axis->storage_size();
  }

  _has_state = _subaxes.count("state");
  _has_old_state = _subaxes.count("old_state");
  _has_forces = _subaxes.count("forces");
  _has_old_forces = _subaxes.count("old_forces");
  _has_residual = _subaxes.count("residual");
  _has_parameters = _subaxes.count("parameters");
}

size_t
LabeledAxis::nvariable(bool recursive) const
{
  return variable_names(recursive).size();
}

size_t
LabeledAxis::nsubaxis(bool recursive) const
{
  return subaxis_names(recursive).size();
}

bool
LabeledAxis::has_variable(const LabeledAxisAccessor & var) const
{
  if (var.empty())
    return false;

  if (var.vec().size() > 1)
  {
    if (has_subaxis(var.vec()[0]))
      return subaxis(var.vec()[0]).has_variable(var.slice(1));
    return false;
  }

  return _variables.count(var.vec()[0]);
}

bool
LabeledAxis::has_subaxis(const LabeledAxisAccessor & s) const
{
  if (s.empty())
    return false;

  if (s.vec().size() > 1)
  {
    if (has_subaxis(s.vec()[0]))
      return subaxis(s.vec()[0]).has_subaxis(s.slice(1));
    return false;
  }

  return _subaxes.count(s.vec()[0]);
}

Size
LabeledAxis::storage_size(const LabeledAxisAccessor & name) const
{
  if (name.empty())
    return _offset;

  if (name.size() == 1)
  {
    if (_variables.count(name.vec()[0]))
      return _variables.at(name.vec()[0]);

    if (_subaxes.count(name.vec()[0]))
      return _subaxes.at(name.vec()[0])->storage_size();

    neml_assert_dbg(false, "Trying to find the storage size of a non-existent item named ", name);
  }

  return subaxis(name.vec()[0]).storage_size(name.slice(1));
}

indexing::TensorIndex
LabeledAxis::indices(const LabeledAxisAccessor & accessor) const
{
  if (accessor.empty())
    return torch::indexing::Slice();

  return indices(0, accessor.begin(), accessor.end());
}

indexing::TensorIndex
LabeledAxis::indices(Size offset,
                     const LabeledAxisAccessor::const_iterator & cur,
                     const LabeledAxisAccessor::const_iterator & end) const
{
  neml_assert_dbg(_layout.count(*cur), "Axis/variable named ", *cur, " does not exist.");
  const auto & [rbegin, rend] = _layout.at(*cur);
  if (cur == end - 1)
    return torch::indexing::Slice(offset + rbegin, offset + rend);

  return subaxis(*cur).indices(offset + rbegin, cur + 1, end);
}

std::vector<std::pair<indexing::TensorIndex, indexing::TensorIndex>>
LabeledAxis::common_indices(const LabeledAxis & other, bool recursive) const
{
  std::vector<std::pair<indexing::TensorIndex, indexing::TensorIndex>> indices;
  std::vector<Size> idxa;
  std::vector<Size> idxb;
  common_indices(other, recursive, idxa, idxb, 0, 0);

  if (idxa.empty())
    return indices;

  // We could be smart and merge contiguous indices
  size_t i = 0;
  size_t j = 1;
  while (j < idxa.size() - 1)
  {
    if (idxa[j] == idxa[j + 1] && idxb[j] == idxb[j + 1])
      j += 2;
    else
    {
      indices.push_back({indexing::Slice(idxa[i], idxa[j]), indexing::Slice(idxb[i], idxb[j])});
      i = j + 1;
      j = i + 1;
    }
  }
  indices.push_back({indexing::Slice(idxa[i], idxa[j]), indexing::Slice(idxb[i], idxb[j])});

  return indices;
}

void
LabeledAxis::common_indices(const LabeledAxis & other,
                            bool recursive,
                            std::vector<Size> & idxa,
                            std::vector<Size> & idxb,
                            Size offseta,
                            Size offsetb) const
{
  for (const auto & [name, sz] : _variables)
    if (other.has_variable(name))
    {
      auto && [begina, enda] = _layout.at(name);
      idxa.push_back(offseta + begina);
      idxa.push_back(offseta + enda);
      auto && [beginb, endb] = other._layout.at(name);
      idxb.push_back(offsetb + beginb);
      idxb.push_back(offsetb + endb);
    }

  if (recursive)
    for (const auto & [name, axis] : _subaxes)
      if (other.has_subaxis(name))
        axis->common_indices(other.subaxis(name),
                             true,
                             idxa,
                             idxb,
                             offseta + _layout.at(name).first,
                             offsetb + other._layout.at(name).first);
}

std::vector<LabeledAxisAccessor>
LabeledAxis::sort_by_assembly_order(const std::set<LabeledAxisAccessor> & names) const
{
  neml_assert(_offset > 0, "The LabeledAxis either is empty or has not been setup");

  std::map<indexing::TensorIndex, const LabeledAxisAccessor &, AssemblySliceCmp> index_names;
  for (const auto & name : names)
    index_names.emplace(indices(name), name);

  std::vector<LabeledAxisAccessor> sorted;
  for (const auto & [idx, name] : index_names)
    sorted.push_back(name);

  return sorted;
}

std::set<LabeledAxisAccessor>
LabeledAxis::variable_names(bool recursive) const
{
  std::set<LabeledAxisAccessor> accessors;

  // Insert local variables
  for (const auto & [var, sz] : _variables)
    accessors.insert(var);

  // Insert variables on subaxes
  if (recursive)
    for (const auto & [name, axis] : _subaxes)
      for (const auto & var : axis->variable_names(true))
        accessors.insert(var.prepend(name));

  return accessors;
}

std::set<LabeledAxisAccessor>
LabeledAxis::subaxis_names(bool recursive) const
{
  std::set<LabeledAxisAccessor> accessors;

  for (const auto & [name, axis] : _subaxes)
  {
    // Insert local subaxes
    accessors.insert(name);
    // Insert sub-subaxes
    if (recursive)
      for (const auto & subname : axis->subaxis_names(true))
        accessors.insert(subname.prepend(name));
  }

  return accessors;
}

const LabeledAxis &
LabeledAxis::subaxis(const LabeledAxisAccessor & name) const
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<LabeledAxis *>(this)->subaxis(name);
}

LabeledAxis &
LabeledAxis::subaxis(const LabeledAxisAccessor & name)
{
  neml_assert(!name.empty(), "sub-axis name cannot be empty");
  neml_assert_dbg(_subaxes.count(name.vec()[0]),
                  "In LabeledAxis::subaxis, no subaxis matches given name ",
                  name);

  if (name.size() > 1)
    return _subaxes[name.vec()[0]]->subaxis(name.slice(1));

  return *_subaxes[name.vec()[0]];
}

bool
LabeledAxis::equals(const LabeledAxis & other) const
{
  // They must have the same size
  if (_offset != other._offset)
    return false;

  // Comparing unordered maps is easy, two maps are equal if they have the same number of
  // elements and the elements in one container are a permutation of the elements in the other
  // container
  if (_variables != other._variables)
    return false;

  // For subaxes, it's a little bit tricky as we need to compare the dereferenced axes.
  for (const auto & [name, axis] : _subaxes)
  {
    if (other._subaxes.count(name) == 0)
      return false;

    if (*other._subaxes.at(name) != *axis)
      return false;
  }

  return true;
}

std::ostream &
operator<<(std::ostream & os, const LabeledAxis & axis)
{
  // Collect variable names and indices
  size_t max_var_name_length = 0;
  std::map<std::string, indexing::TensorIndex> vars;
  for (auto var : axis.variable_names(true))
  {
    auto var_name = utils::stringify(var);
    if (var_name.size() > max_var_name_length)
      max_var_name_length = var_name.size();
    vars.emplace(var_name, axis.indices(var));
  }

  // Print variables with right alignment
  for (auto var = vars.begin(); var != vars.end(); var++)
  {
    // NOLINTNEXTLINE(*-narrowing-conversions)
    os << std::setw(max_var_name_length) << var->first << ": " << var->second;
    if (std::next(var) != vars.end())
      os << std::endl;
  }

  return os;
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
