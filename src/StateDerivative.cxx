#include "StateDerivative.h"
#include "SymSymR4.h"

using namespace torch::indexing;

StateDerivative::StateDerivative(const StateInfo & A, const StateInfo & B, TorchSize nbatch)
  : StateBase(torch::zeros(add_shapes({nbatch}, make_shape(A, B)), TorchDefaults)),
    _A(A),
    _B(B)
{
  setup_views();
}

StateDerivative::StateDerivative(const StateInfo & A,
                                 const StateInfo & B,
                                 const torch::Tensor & tensor)
  : StateBase(tensor),
    _A(A),
    _B(B)
{
  // Check that the size of the tensor was compatible
  if (tensor.sizes() != add_shapes({tensor.sizes()[0]}, make_shape(A, B)))
    throw std::runtime_error("Tensor provided to StateDerivative does not "
                             "have the right size to hold the derivative "
                             "of State A with respect to State B");
  setup_views();
}

StateDerivative::StateDerivative(const State & A, const State & B)
  : StateBase(
        torch::zeros(add_shapes({A.batch_size()}, make_shape(A.info(), B.info())), TorchDefaults)),
    _A(A.info()),
    _B(B.info())
{
  // Check that the two batch sizes were consistent
  if (A.batch_size() != B.batch_size())
    throw std::runtime_error("The batch sizes of State A and State B are "
                             "not consistent.");

  setup_views();
}

StateDerivative
StateDerivative::id_map(const StateInfo & A,
                        const StateInfo & B,
                        TorchSize nbatch,
                        std::map<std::string, std::string> id_map)
{
  StateDerivative map = StateDerivative(A, B, nbatch);

  for (auto name : A.items())
    map.set(name,
            id_map.at(name),
            torch::repeat_interleave(
                torch::eye(A.base_storage(name), TorchDefaults).unsqueeze(0), nbatch, 0));

  return map;
}

StateDerivative
StateDerivative::same_batch(const StateInfo & A, const State & B)
{
  return StateDerivative(A, B.info(), B.batch_size());
}

StateDerivative
StateDerivative::promote(std::string left, std::string right, SymSymR4 C)
{
  StateInfo A;
  A.add<SymR2>(left);
  StateInfo B;
  B.add<SymR2>(right);
  return StateDerivative(A, B, C);
}

const StateInfo &
StateDerivative::info_A() const
{
  return _A;
}

const StateInfo &
StateDerivative::info_B() const
{
  return _B;
}

StateDerivative
StateDerivative::get_substate(std::string name_A, std::string name_B)
{
  return StateDerivative(_A.substates().at(name_A),
                         _B.substates().at(name_B),
                         (*this)[derivative_name(name_A, name_B)]);
}

std::string
StateDerivative::derivative_name(std::string name_A, std::string name_B)
{
  // Compiler should make this concatenation, but check later
  return "∂(" + name_A + ")/∂(" + name_B + ")";
}

StateDerivative
StateDerivative::chain(const StateDerivative & other) const
{
  // This function expresses a chain rule, which is just a dot
  // product between the values of this and the values of the input
  // The main annoyance is just getting the names correct

  // Check that we are conformal
  if (batch_size() != other.batch_size())
    throw std::runtime_error("StateDerivative batch sizes are "
                             "not the same");
  if (_B != other.info_A())
    throw std::runtime_error("StateInfo objects are not conformal");

  // If all the sizes are correct then executing the chain rule is
  // pretty easy
  return StateDerivative(_A, other.info_B(), torch::bmm(tensor(), other.tensor()));
}

StateDerivative
StateDerivative::slice_left(std::string group) const
{
  return StateDerivative(
      _A.substates().at(group), _B, tensor().index({Slice(), _A_groups.at(group), Slice()}));
}

void
StateDerivative::set_slice(std::string group, StateDerivative other)
{
  tensor().index_put_({Slice(), _A_groups.at(group), Slice()}, other.tensor());
}

StateDerivative
StateDerivative::slice_right(std::string group) const
{
  return StateDerivative(
      _A, _B.substates().at(group), tensor().index({Slice(), Slice(), _B_groups.at(group)}));
}

StateDerivative
StateDerivative::scalar_product(const Scalar & other) const
{
  return StateDerivative(_A, _B, tensor() * other.unsqueeze(-1));
}

StateDerivative
StateDerivative::replace_info_left(const StateInfo & input) const
{
  return StateDerivative(input, _B, tensor());
}

StateDerivative
StateDerivative::replace_info_right(const StateInfo & input) const
{
  return StateDerivative(_A, input, tensor());
}

StateDerivative
StateDerivative::operator-() const
{
  return StateDerivative(_A, _B, -tensor().clone());
}

StateDerivative
StateDerivative::operator+=(const StateDerivative & other) const
{
  // Check that we are conformal
  if (batch_size() != other.batch_size())
    throw std::runtime_error("StateDerivative batch sizes are "
                             "not the same");
  if ((_A != other.info_A()) || (_B != other.info_B()))
    throw std::runtime_error("StateInfo objects are not conformal");

  return StateDerivative(_A, _B, tensor().add(other.tensor()));
}

StateDerivative
StateDerivative::inverse() const
{
  return StateDerivative(_B, _A, torch::linalg::inv(tensor()));
}

StateDerivative
StateDerivative::add_identity() const
{
  // Make debug
  if (_A.size_storage() != _B.size_storage())
    throw std::runtime_error("Can only add identity to square derivatives");

  return StateDerivative(
      _A,
      _B,
      tensor() + torch::repeat_interleave(
                     torch::eye(_A.size_storage(), TorchDefaults).unsqueeze(0), batch_size(), 0));
}

TorchShape
StateDerivative::make_shape(const StateInfo & A, const StateInfo & B)
{
  return {A.size_storage(), B.size_storage()};
}

void
StateDerivative::setup_views()
{
  // The objects themselves are pretty easy, just assemble names
  // and views by concatenating the two states.
  for (auto A : _A.item_locations())
  {
    for (auto B : _B.item_locations())
    {
      add_label(derivative_name(A.first, B.first),
                {Slice(_A.item_offsets()[A.second], _A.item_offsets()[A.second + 1]),
                 Slice(_B.item_offsets()[B.second], _B.item_offsets()[B.second + 1])});
    }
  }

  // Actually pretty similar for substates, just with the added
  // annoyance in accessing the substate info objects
  for (auto A : _A.substate_locations())
  {
    for (auto B : _B.substate_locations())
    {
      auto sub_A_info = _A.substates().at(A.first);
      auto sub_B_info = _B.substates().at(B.first);
      std::string name = derivative_name(A.first, B.first);
      _A_groups.insert(
          {A.first,
           Slice(_A.item_offsets()[A.second], _A.item_offsets()[A.second + sub_A_info.nitems()])});
      _B_groups.insert(
          {B.first,
           Slice(_B.item_offsets()[B.second], _B.item_offsets()[B.second + sub_B_info.nitems()])});

      add_label(name, {_A_groups.at(A.first), _B_groups.at(B.first)});
    }
  }
}

StateDerivative
operator+(const StateDerivative & A, const StateDerivative & B)
{
  return A += B;
}
