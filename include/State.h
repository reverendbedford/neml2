#pragma once

#include "StandardBatchedLabeledTensor.h"
#include "StateInfo.h"

/// A state object providing flat tensor storage and easy access for model state
class State : public StandardBatchedLabeledTensor
{
 public:
  /// Setup with new storage
  State(const StateInfo & info, TorchSize nbatch);

  /// Setup with existing storage
  State(const StateInfo & info, const torch::Tensor & tensor);

  /// Helper to return the batch size
  TorchSize batch_size() const;

  template<typename T>
  struct item_type{ typedef T type; };

  /// As view but also interpret the view as an object
  template <typename T>
  typename item_type<T>::type get(std::string name)
  {
    // Ugh have to do the shape dynamically because the batch
    // size is not yet fixed
    TorchShape base({batch_size()});
    base.insert(base.end(), T::base_shape.begin(), T::base_shape.end());
    return T(view(name).view(base));
  }

  /// Used to differentiate slices of substates from slices of objects
  inline static const std::string substate_prefix = "substate_";

 protected:
  /// Actually do the work of setting up all the required views
  void setup_views();

 protected:
  StateInfo _info;
};
