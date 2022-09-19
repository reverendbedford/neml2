#include "StateDerivative.h"

using namespace torch::indexing;

StateDerivative::StateDerivative(const StateInfo & A, const StateInfo & B,
                                 TorchSize nbatch) :
    StateBase(torch::empty(add_shapes({nbatch}, make_shape(A,B)))), 
    _A(A), _B(B)
{
  setup_views();
}

StateDerivative::StateDerivative(const StateInfo & A, const StateInfo & B,
                                 const torch::Tensor & tensor) :
    StateBase(tensor), _A(A), _B(B)
{
  // Check that the size of the tensor was compatible
  if (sizes() != make_shape(A, B))
    throw std::runtime_error("Tensor provided to StateDerivative does not "
                             "have the right size to hold the derivative "
                             "of State A with respect to State B");
  setup_views();
}

StateDerivative::StateDerivative(const State & A, const State & B) :
    StateBase(torch::empty(add_shapes({A.batch_size()}, 
                                      make_shape(A.info(), B.info())))),
    _A(A.info()), _B(B.info())
{
  // Check that the two batch sizes were consistent
  if (A.batch_size() != B.batch_size())
    throw std::runtime_error("The batch sizes of State A and State B are "
                             "not consistent.");

  setup_views();
}

const StateInfo & StateDerivative::info_A() const
{
  return _A;
}

const StateInfo & StateDerivative::info_B() const
{
  return _B;
}

StateDerivative StateDerivative::get_substate(std::string name_A, 
                                              std::string name_B)
{
  return StateDerivative(_A.substates().at(name_A), 
                         _B.substates().at(name_B),
                         get_view(derivative_name(name_A, name_B)));
}

std::string StateDerivative::derivative_name(std::string name_A, 
                                             std::string name_B)
{
  // Compiler should make this concatenation, but check later
  return "d_" + name_A + "_d_" + name_B;
}

TorchShape StateDerivative::make_shape(const StateInfo & A,
                                       const StateInfo & B)
{
  return {A.size_storage(), B.size_storage()};
}

void StateDerivative::setup_views() 
{
  // The objects themselves are pretty easy, just assemble names
  // and views by concatenating the two states.
  for (auto A : _A.item_locations()) {
    for (auto B : _B.item_locations()) {
      add_label(derivative_name(A.first, B.first), 
                {Slice(_A.item_offsets()[A.second],
                       _A.item_offsets()[A.second+1]),
                Slice(_B.item_offsets()[B.second],
                      _B.item_offsets()[B.second+1])});
    }
  }

  // Actually pretty similar for substates, just with the added 
  // annoyance in accessing the substate info objects
  for (auto A : _A.substate_locations()) {
    for (auto B: _B.substate_locations()) {
      auto sub_A_info = _A.substates().at(A.first);
      auto sub_B_info = _B.substates().at(B.first);
      std::string name = derivative_name(A.first, B.first);
      add_label(name,
                {Slice(_A.item_offsets()[A.second],
                       _A.item_offsets()[A.second + 
                       sub_A_info.nitems()]),
                Slice(_B.item_offsets()[B.second],
                      _B.item_offsets()[B.second + 
                      sub_B_info.nitems()])});
    }
  }
}
