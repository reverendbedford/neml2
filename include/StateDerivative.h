#pragma once

#include "StateBase.h"
#include "State.h"
#include "StateInfo.h"

/// A logically 2D (nbatch,m,n) tensor with labels storing a state derivative
class StateDerivative : public StateBase
{
 public:
  /// Construct with storage
  StateDerivative(const StateInfo & A, const StateInfo & B, 
                  TorchSize nbatch);
  /// Construct with external storage
  StateDerivative(const StateInfo & A, const StateInfo & B,
                  const torch::Tensor & tensor);
  /// Construct from two already-setup states (can infer batch)
  StateDerivative(const State & A, const State & B);

  /// Helper to return the information on the base state
  const StateInfo & info_A() const;
  
  /// Helper to return the information on the derivative state
  const StateInfo & info_B() const;

  template<typename T>
  struct item_type{ typedef T type; };

  /// As view but also interpret the view as an object
  template <typename T>
  typename item_type<T>::type get(std::string name_A, std::string name_B)
  {
    // Could partly be transferred to base class
    return T(get_view(derivative_name(name_A, name_B)).view(
            add_shapes({batch_size()}, T::base_shape)));
  }

  /// Set a object value into a spot in the object
  template <typename T>
  void set(std::string name_A, std::string name_B, const T & value)
  {
    // Shape should be (batch,storage_A,storage_B)
    set_view(derivative_name(name_A, name_B), 
             value.reshape({
                           batch_size(),
                           _A.base_storage(name_A),
                           _B.base_storage(name_B)}));
  }

  /// No reshape required and special logic to setup
  StateDerivative get_substate(std::string name_A, std::string name_B);

  /// Naming convention for the derivative
  static std::string derivative_name(std::string name_A, std::string name_B);

 protected:
  static TorchShape make_shape(const StateInfo & A, const StateInfo & B);
  void setup_views();

 protected:
  StateInfo _A, _B;
};
