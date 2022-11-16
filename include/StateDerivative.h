#pragma once

#include "State.h"
#include "StateBase.h"
#include "StateInfo.h"

using namespace torch::indexing;

class SymSymR4;

// Forward declaration
class State;

/// A logically 2D (nbatch,m,n) tensor with labels storing a state derivative
class StateDerivative : public StateBase
{
public:
  /// Debate adding a default constructor
  virtual ~StateDerivative(){};

  /// Construct with storage
  StateDerivative(const StateInfo & A, const StateInfo & B, TorchSize nbatch);
  /// Construct with external storage
  StateDerivative(const StateInfo & A, const StateInfo & B, const torch::Tensor & tensor);
  /// Construct from two already-setup states (can infer batch)
  StateDerivative(const State & A, const State & B);

  /// Make a block identity tensor mapping one state into another
  static StateDerivative id_map(const StateInfo & A,
                                const StateInfo & B,
                                TorchSize nbatch,
                                std::map<std::string, std::string> id_map);

  /// Shortcut to create with batch from another state
  static StateDerivative same_batch(const StateInfo & A, const State & B);

  // Promote shortcuts, helpful in forming chain rules
  static StateDerivative promote(std::string left, std::string right, SymSymR4 C);

  /// Helper to return the information on the base state
  const StateInfo & info_A() const;

  /// Helper to return the information on the derivative state
  const StateInfo & info_B() const;

  template <typename T>
  struct item_type
  {
    typedef T type;
  };

  /// As view but also interpret the view as an object
  template <typename T>
  typename item_type<T>::type get(std::string name_A, std::string name_B)
  {
    // Could partly be transferred to base class
    return T(
        (*this)[derivative_name(name_A, name_B)].view(add_shapes({batch_size()}, T::_base_sizes)));
  }

  /// Set a object value into a spot in the object
  template <typename T>
  void set(std::string name_A, std::string name_B, const T & value)
  {
    // Shape should be (batch,storage_A,storage_B)
    (*this)[derivative_name(name_A, name_B)].index_put_(
        {None}, value.reshape({batch_size(), _A.base_storage(name_A), _B.base_storage(name_B)}));
  }

  /// No reshape required and special logic to setup
  StateDerivative get_substate(std::string name_A, std::string name_B);

  /// Naming convention for the derivative
  static std::string derivative_name(std::string name_A, std::string name_B);

  /// Chain rule product of two derivatives
  StateDerivative chain(const StateDerivative & other) const;

  /// Make a new StateDerivative with only a single substate on the left
  StateDerivative slice_left(std::string group) const;

  /// Set an entire group of rows
  void set_slice(std::string group, StateDerivative other);

  /// Make a new StateDerivative with only a single substate on the right
  StateDerivative slice_right(std::string group) const;

  /// Scalar product
  //  This only exists because we have to do YieldSurfaces outside of the
  //  State and StateFunction system.  If we had proper second derivatives
  //  we would not need this and could just use chain...
  StateDerivative scalar_product(const Scalar & other) const;

  /// Replace the "left" state
  StateDerivative replace_info_left(const StateInfo & input) const;

  /// Replace the "right" state
  StateDerivative replace_info_right(const StateInfo & input) const;

  /// Unary minus = additive inverse
  StateDerivative operator-() const;

  /// Add two StateDerivative objects together to complete a chain rule
  StateDerivative operator+=(const StateDerivative & other) const;

  /// Invert a StateDerivative for use in an implicit function derivative
  StateDerivative inverse() const;

  /// Add the identity, which is used in some implicit integration routines
  StateDerivative add_identity() const;

protected:
  static TorchShape make_shape(const StateInfo & A, const StateInfo & B);
  void setup_views();

protected:
  StateInfo _A, _B;
  std::map<std::string, TorchIndex> _A_groups;
  std::map<std::string, TorchIndex> _B_groups;
};

/// StateDerivative chain rule addition
StateDerivative operator+(const StateDerivative & A, const StateDerivative & B);
