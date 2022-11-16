#pragma once

#include <map>
#include <type_traits>
#include "types.h"
#include "Scalar.h"
#include "SymR2.h"

/// The list of all objects currently allowed in a state object
template <typename T>
struct is_state : std::false_type
{
};
template <>
struct is_state<Scalar> : std::true_type
{
};
template <>
struct is_state<SymR2> : std::true_type
{
};

/// Class defining basic information about the state of a model
//  This class has all the information you need to setup a State
//  object but does not actually store data
class StateInfo
{
public:
  StateInfo();
  StateInfo(std::map<std::string, size_t> locations, std::vector<TorchSize> offsets);

  /// Add any tensor that can be stored in a State object
  template <typename T, typename = std::enable_if_t<is_state<T>::value>>
  void add(std::string name)
  {
    // The storage is *flat* -- will need to reshape when we return!
    auto sz = storage_size(T::_base_sizes);
    add(name, sz);
  }

  /// Add some substate defined by another StateInfo object
  void add_substate(std::string name, const StateInfo & substate);

  /// Number of items stored
  size_t nitems() const;

  /// Total size of storage required
  TorchSize size_storage() const;

  /// Getter for item locations
  const std::map<std::string, size_t> & item_locations() const { return _item_locations; }

  /// Getter for item offsets
  const std::vector<TorchSize> & item_offsets() const { return _item_offsets; }

  /// Number of substate items
  size_t nsubstates() const;

  /// Getter for substate locations
  const std::map<std::string, size_t> & substate_locations() const { return _substate_locations; }

  /// Getter for substate objects
  const std::map<std::string, StateInfo> & substates() const { return _substates; }

  /// Helper to report the shape of the tensor needed, given the batch size
  TorchShape required_shape(TorchSize nbatch) const;

  /// Base storage (ignoring batch) needed for an object
  TorchSize base_storage(std::string object) const;

  /// Add a prefix to the names of the items
  StateInfo & add_prefix(std::string prefix);

  /// Add a suffix to the names of the items
  StateInfo & add_suffix(std::string suffix);

  /// Check to see if two StateInfo objects are equivalent
  bool equals(const StateInfo & other) const;

  /// Just get the names of the variables
  std::vector<std::string> items() const;

  /// Just get the names of the substate objects
  std::vector<std::string> substate_names() const;

  /// Rename a particular item
  StateInfo & rename(std::string original, std::string rename);

  /// Return StateInfo with one item removed
  StateInfo remove(std::string name) const;

  /// Add an item from raw name/storage information
  void add(std::string name, size_t sz);

protected:
  // Location of each item in the vector of offsets
  std::map<std::string, size_t> _item_locations;

  // Offset into storage for each item
  std::vector<TorchSize> _item_offsets;

  // Start of each substate as an item_location
  std::map<std::string, size_t> _substate_locations;

  // The full substate items, stored for convenience in reconstructing
  std::map<std::string, StateInfo> _substates;
};

/// Equality between StateInfo objects, expressed via the equals method
bool operator==(const StateInfo & a, const StateInfo & b);

/// Inequality between StateInfo objects
bool operator!=(const StateInfo & a, const StateInfo & b);
