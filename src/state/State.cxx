#include "state/State.h"
#include "tensors/Scalar.h"

using namespace torch::indexing;

State::State(const StateInfo & info, TorchSize nbatch)
  : StateBase(torch::zeros(info.required_shape(nbatch), TorchDefaults)),
    _info(info)
{
  setup_views();
}

State::State(const StateInfo & info, const torch::Tensor & tensor)
  : StateBase(tensor),
    _info(info)
{
  if (tensor.sizes() != _info.required_shape(tensor.sizes()[0]))
    throw std::runtime_error("Provided tensor does not have the correct shape "
                             "for the state described by the provided "
                             "StateInfo");
  setup_views();
}

State
State::same_batch(const StateInfo & info, const State & other)
{
  return State(info, other.batch_size());
}

State
State::clone() const
{
  auto nstate = State(_info, batch_size());
  nstate.tensor().copy_(tensor());
  return nstate;
}

State
State::get_substate(std::string name)
{
  // No reshaping is needed, but we do need to grab the
  // appropriate StateInfo
  return State(_info.substates().at(name), (*this)[name]);
}

void
State::set_substate(std::string name, State substate)
{
  (*this)[name].index_put_({None}, substate.tensor());
}

const StateInfo &
State::info() const
{
  return _info;
}

void
State::setup_views()
{
  // Just add the items one by one as views into the flat array
  for (auto item : _info.item_locations())
  {
    add_label(item.first,
              {Slice(_info.item_offsets()[item.second], _info.item_offsets()[item.second + 1])});
  }

  // Can be done exactly the same as the actual objects
  for (auto item : _info.substate_locations())
  {
    auto substate_info = _info.substates().at(item.first);
    add_label(item.first,
              {Slice(_info.item_offsets()[item.second],
                     _info.item_offsets()[item.second + substate_info.nitems()])});
  }
}

State &
State::rename(std::string original, std::string rename)
{
  _info.rename(original, rename);
  return *this;
}

StateDerivative
State::promote_left(std::string scalar_name)
{
  StateInfo A;
  A.add<Scalar>(scalar_name);

  return StateDerivative(A, _info, tensor().unsqueeze(1));
}

StateDerivative
State::promote_right(std::string scalar_name)
{
  StateInfo B;
  B.add<Scalar>(scalar_name);

  return StateDerivative(_info, B, tensor().unsqueeze(-1));
}

StateDerivative
State::promote_outer(State B) const
{
  return StateDerivative(
      _info, B.info(), torch::bmm(tensor().unsqueeze(2), B.tensor().unsqueeze(1)));
}

State
State::scalar_product(Scalar scalar) const
{
  return State(_info, tensor() * scalar);
}

State
State::replace_info(const StateInfo & info) const
{
  return State(info, tensor());
}

State
State::add(State state) const
{
  // Make debug
  if (!_info.equals(state.info()))
    throw std::runtime_error("StateInfo for states being added must be "
                             "the same");

  return State(_info, tensor() + state.tensor());
}

State
State::subtract(State state) const
{
  // Make debug
  if (!_info.equals(state.info()))
    throw std::runtime_error("StateInfo for states being added must be "
                             "the same");

  return State(_info, tensor() - state.tensor());
}

State
State::remove(std::string item) const
{
  // Get a view into the data to all the items except the
  // one we're trying to remove
  std::vector<TorchSize> inds;
  for (auto it : _info.items())
  {
    if (it == item)
      continue;

    TorchSize of = _info.item_offsets()[_info.item_locations().at(it)];
    TorchSize sz = _info.base_storage(it);
    for (TorchSize i = 0; i < sz; i++)
      inds.push_back(i + of);
  }
  auto it_ten = torch::from_blob(inds.data(),
                                 {static_cast<TorchSize>(inds.size())},
                                 torch::TensorOptions().dtype(torch::kInt64));

  // Setup with a new State with the StateInfo updated
  return State(_info.remove(item), tensor().index({Slice(), it_ten}));
}

State
State::add_suffix(std::string suffix)
{
  return State(_info.add_suffix(suffix), tensor());
}
