// Copyright 2023, UChicago Argonne, LLC
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

#pragma once

#include <unordered_map>
#include <type_traits>

#include "neml2/misc/types.h"

#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/SymR2.h"

namespace neml2
{
template <typename T>
struct is_labelable : std::false_type
{
};
template <>
struct is_labelable<Scalar> : std::true_type
{
};
template <>
struct is_labelable<SymR2> : std::true_type
{
};

/**
The accessor containing all the information needed to access an item in a `LabeledAxis`. The
accessor consists of an arbitrary number of item names.
The last item name can be either a variable name or a sub-axis name.
All the other item names are considered to be sub-axis names.
*/
struct LabeledAxisAccessor
{
  std::vector<std::string> item_names;

  operator std::vector<std::string>() const { return item_names; }

  bool empty() const { return item_names.empty(); }

  LabeledAxisAccessor with_suffix(const std::string & suffix) const
  {
    auto new_names = item_names;
    new_names.back() += suffix;
    return LabeledAxisAccessor{new_names};
  }

  LabeledAxisAccessor on(const std::string & axis) const
  {
    auto new_names = item_names;
    new_names.insert(new_names.begin(), axis);
    return LabeledAxisAccessor{new_names};
  }

  LabeledAxisAccessor on(const LabeledAxisAccessor & axis) const
  {
    auto new_names = axis.item_names;
    new_names.insert(new_names.end(), item_names.begin(), item_names.end());
    return LabeledAxisAccessor{new_names};
  }

  bool operator==(const LabeledAxisAccessor & other) const
  {
    return item_names == other.item_names;
  }

  bool operator<(const LabeledAxisAccessor & other) const { return item_names < other.item_names; }

  friend std::ostream & operator<<(std::ostream & os, const LabeledAxisAccessor & accessor);
};

std::ostream & operator<<(std::ostream & os, const LabeledAxisAccessor & accessor);

class LabeledAxis;

typedef std::unordered_map<std::string, std::pair<TorchSize, TorchSize>> AxisLayout;

class LabeledAxis
{
public:
  LabeledAxis();

  /// (Shallow) copy constructor
  LabeledAxis(const LabeledAxis & other);

  /**
   * All the LabeledAxis modifiers. The modifiers can only be used during the setup stage. Calling
   * any modifiers after the setup stage is forbidden and will result in a runtime error in Debug
   * mode.
   *
   * All the modifiers return the modified LabeledAxis by reference, so modifiers can be chained.
   For
   * example
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
   LabeledAxis labels = input().clone();
   labels.add<SymR2>("stress").rename("stress", "stress1").remove("stress1");
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   */
  /// @{
  /// Add a variable or subaxis
  template <typename T>
  LabeledAxis & add(const std::string & name)
  {
    // The storage is *flat* -- will need to reshape when we return!
    // All NEML2 primitive data types will have the member _base_sizes
    if constexpr (is_labelable<T>::value)
    {
      auto sz = utils::storage_size(T::_base_sizes);
      add(name, sz);
      return *this;
    }

    // Add an empty subaxis
    if constexpr (std::is_same_v<LabeledAxis, T>)
    {
      if (!has_subaxis(name))
        _subaxes.emplace(name, std::make_shared<LabeledAxis>());
      return *this;
    }
  }

  /// Add an arbitrary variable with known storage size.
  LabeledAxis & add(const std::string & name, TorchSize sz);

  /// Add an arbitrary variable using a `LabeledAxisAccessor`
  LabeledAxis & add(const LabeledAxisAccessor & accessor, TorchSize sz);

  /// Add prefix to the labels.
  /// Note that this method doesn't recurse through sub-axes.
  LabeledAxis & prefix(const std::string & prefix, const std::string & delimiter = "_");

  /// Add suffix to the labels.
  /// Note that this method doesn't recurse through sub-axes.
  LabeledAxis & suffix(const std::string & suffix, const std::string & delimiter = "_");

  /// Change the label of an item
  LabeledAxis & rename(const std::string & original, const std::string & rename);

  /// Remove an item
  LabeledAxis & remove(const std::string & name);

  /// Clear everything
  LabeledAxis & clear();
  /// @}

  /// Merge with another `LabeledAxis`.
  std::vector<LabeledAxisAccessor> merge(LabeledAxis & other);

  /**
  Setup the layout of all items recursively. The layout of each item is contiguous in memory.
  */
  void setup_layout();

  /// Number of items
  size_t nitem() const { return nvariable() + nsubaxis(); }

  /// Number of variables
  size_t nvariable() const { return _variables.size(); }

  /// Number of subaxes
  size_t nsubaxis() const { return _subaxes.size(); }

  /// Does the item exist?
  bool has_item(const std::string & name) const { return has_variable(name) || has_subaxis(name); }

  /// Does the variable of a given primitive type exist?
  template <typename T>
  bool has_variable(const std::string & name) const
  {
    if constexpr (is_labelable<T>::value)
    {
      if (!has_variable(name))
        return false;

      // also check size
      auto sz = utils::storage_size(T::_base_sizes);
      return _variables.at(name) == sz;
    }
  }

  /// Does the variable exist?
  bool has_variable(const std::string & name) const { return _variables.count(name); }

  /// Does the item exist?
  bool has_subaxis(const std::string & name) const { return _subaxes.count(name); }

  /// (total) storage size
  /// @{
  TorchSize storage_size() const { return _offset; }
  TorchSize storage_size(const std::string &) const;
  TorchSize storage_size(const LabeledAxisAccessor & accessor) const;
  /// @}

  /// Get the layout
  const AxisLayout & layout() const { return _layout; }

  /// @{
  /// Get the indices of a specific item by its name
  TorchIndex indices(const std::string & name) const;
  /// Get the indices of a specific item by a `LabeledAxisAccessor`
  TorchIndex indices(const LabeledAxisAccessor & accessor) const;
  /// Get the indices using another `LabeledAxis`.
  TorchIndex indices(const LabeledAxis & other, bool recursive = true, bool inclusive = true) const;
  /// @}

  /// Get the common indices of two `LabeledAxis`s
  static std::vector<std::pair<TorchIndex, TorchIndex>>
  common_indices(const LabeledAxis & a, const LabeledAxis & b, bool recursive = true);

  /// Get the item names
  std::vector<std::string> item_names() const;

  /// Get the variables
  const std::map<std::string, TorchSize> & variables() const { return _variables; }

  /// Get the subaxes
  const std::map<std::string, std::shared_ptr<LabeledAxis>> & subaxes() const { return _subaxes; }

  /// Get the variable accessors
  std::vector<LabeledAxisAccessor> variable_accessors(bool recursive = false) const;

  /// Get a sub-axis
  /// @{
  const LabeledAxis & subaxis(const std::string & name) const;
  LabeledAxis & subaxis(const std::string & name);
  /// @}

  /// Check to see if two LabeledAxis objects are equivalent
  bool equals(const LabeledAxis & other) const;

  friend std::ostream & operator<<(std::ostream & os, const LabeledAxis & info);

  /// Write this object in dot format
  void to_dot(std::ostream & os,
              int & id,
              std::string name = "",
              bool subgraph = false,
              bool node_handle = false) const;

  /// Helper static variable to keep track of indentation when printing
  static int level;

private:
  void add(LabeledAxis & axis,
           TorchSize sz,
           const std::vector<std::string>::const_iterator & cur,
           const std::vector<std::string>::const_iterator & end) const;

  void merge(LabeledAxis & other,
             std::vector<std::string> subaxes,
             std::vector<LabeledAxisAccessor> & merged_vars);

  /// Helper method to recursively find the storage size of a variable
  TorchSize storage_size(const std::vector<std::string>::const_iterator & cur,
                         const std::vector<std::string>::const_iterator & end) const;

  /// Helper method to recursively consume the sub-axis names of a `LabeledAxisAccessor` to get the
  /// indices of a variable.
  TorchIndex indices(TorchSize offset,
                     const std::vector<std::string>::const_iterator & cur,
                     const std::vector<std::string>::const_iterator & end) const;

  /// Helper method to (recursively) index this axis using another axis
  void indices(const LabeledAxis & other,
               bool recursive,
               bool inclusive,
               std::vector<TorchSize> & idx,
               TorchSize offset) const;

  /// Get the common indices of two `LabeledAxis`s
  static void common_indices(const LabeledAxis & a,
                             const LabeledAxis & b,
                             bool recursive,
                             std::vector<std::pair<TorchIndex, TorchIndex>> & indices,
                             TorchSize offset_a,
                             TorchSize offset_b);

  void variable_accessors(std::vector<LabeledAxisAccessor> & accessors,
                          LabeledAxisAccessor cur,
                          bool recursive) const;

  /// Variables and their sizes
  std::map<std::string, TorchSize> _variables;

  /// Sub-axes
  // Each sub-axis can contain its own variables and sub-axes
  std::map<std::string, std::shared_ptr<LabeledAxis>> _subaxes;

  /// The layout of this `LabeledAxis`, i.e. the name-to-TorchSlice map
  // After all the `LabeledAxis`s are setup, we need to setup the layout once and only once. This is
  // important for performance considerations, as we need to use the layout to construct many,
  // many LabeledVector and LabeledMatrix at runtime, and so we don't want to waste time on setting
  // up the layout over and over again.
  AxisLayout _layout;

  /// The total storage size of the axis.
  // Similar considerations as `_layout`, i.e., the _offset will be zero during the setup stage,
  // and will have a fixed (hopefully correct) size after the layout have been setup.
  TorchSize _offset;
};

std::ostream & operator<<(std::ostream & os, const LabeledAxis & info);

bool operator==(const LabeledAxis & a, const LabeledAxis & b);

bool operator!=(const LabeledAxis & a, const LabeledAxis & b);
} // namespace neml2
