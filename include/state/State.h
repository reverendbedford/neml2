#pragma once

#include "state/StateBase.h"
#include "state/StateDerivative.h"
#include "state/StateInfo.h"

using namespace torch::indexing;

// Forward declaration
class StateDerivative;

/// A state object providing flat tensor storage and easy access for model state
class State : public StateBase
{
public:
  /// Maybe I should just add a default constructor...
  virtual ~State(){};

  /// Setup with new storage
  State(const StateInfo & info, TorchSize nbatch);

  /// Setup with existing storage
  State(const StateInfo & info, const torch::Tensor & tensor);

  /// Shortcut setup from batch size of another object
  static State same_batch(const StateInfo & info, const State & other);

  /// Clone a copy of this state
  State clone() const;

  /// Template setup for appropriate item types
  template <typename T>
  struct item_type
  {
    typedef T type;
  };

  /// As get_view but also interpret the view as an object
  template <typename T>
  typename item_type<T>::type get(std::string name) const
  {
    // Ugh have to do the shape dynamically because the batch
    // size is not yet fixed
    return T((*this)[name].view(add_shapes({batch_size()}, T::_base_sizes)));
  }

  /// As set_view but also interpret the input as an object
  template <typename T>
  void set(std::string name, const T & value)
  {
    // Need to flatten for the same reason as in get
    (*this)[name].index_put_({None}, value.flatten(1));
  }

  /// No reshape required and special logic to setup
  State get_substate(std::string name);

  /// Special case for setting an entire substate
  void set_substate(std::string name, State substate);

  /// Getter for the information object
  const StateInfo & info() const;

  /// Rename a particular item
  State & rename(std::string original, std::string rename);

  /// Promote to StateDerivative by adding a scalar to the left
  StateDerivative promote_left(std::string scalar_name);

  /// Promote to a StateDerivative by adding a scalar to the right
  StateDerivative promote_right(std::string scalar_name);

  /// Promote to StateDerivative with arbitrary outer product
  StateDerivative promote_outer(State B) const;

  /// Scalar multiplication with a Batched Scalar
  State scalar_product(Scalar scalar) const;

  /// Replace the StateInfo object with another (compatible) one
  State replace_info(const StateInfo & info) const;

  /// Add two state objects
  State add(State other) const;

  /// Subtract two state objects
  State subtract(State other) const;

  /// Remove an object and return everything else
  State remove(std::string item) const;

  /// Alias to StateInfo::add_suffix
  State add_suffix(std::string suffix);

protected:
  /// Actually do the work of setting up all the required views
  virtual void setup_views();

protected:
  StateInfo _info;
};
