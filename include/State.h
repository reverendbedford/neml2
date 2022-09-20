#pragma once

#include "StateBase.h"
#include "StateInfo.h"

using namespace torch::indexing;

/// A state object providing flat tensor storage and easy access for model state
class State : public StateBase
{
 public:
  /// Setup with new storage
  State(const StateInfo & info, TorchSize nbatch);

  /// Setup with existing storage
  State(const StateInfo & info, const torch::Tensor & tensor);

  template<typename T>
  struct item_type{ typedef T type; };

  /// As get_view but also interpret the view as an object
  template <typename T>
  typename item_type<T>::type get(std::string name)
  {
    // Ugh have to do the shape dynamically because the batch
    // size is not yet fixed
    return T(get_view(name).view(add_shapes({batch_size()}, T::base_shape)));
  }

  /// As set_view but also interpret the input as an object
  template <typename T>
  typename item_type<T>::type set(std::string name, const T & value)
  {
    // Need to flatten for the same reason as in get
    set_view(name, value.view({nbatch(), -1}));
  }

  /// No reshape required and special logic to setup
  State get_substate(std::string name);

  /// Getter for the information object
  const StateInfo & info() const;

 protected:
  /// Actually do the work of setting up all the required views
  virtual void setup_views();

 protected:
  StateInfo _info;
};
