#include "models/ADModel.h"

std::tuple<LabeledVector, LabeledMatrix>
ADModel::value_and_dvalue(LabeledVector in) const
{
  bool was_ad = true;
  if (!in.tensor().requires_grad())
  {
    was_ad = false;
    in.tensor().requires_grad_();
  }

  // Evalute the model (not its derivatives)
  auto out = Model::value(in);

  // Allocate space for Jacobian
  LabeledMatrix dout_din(out, in);

  // Loop over rows of the state to retrieve the derivatives
  for (TorchSize i = 0; i < out.tensor().base_sizes()[0]; i++)
  {
    BatchTensor<1> grad_outputs = torch::zeros_like(out.tensor());
    grad_outputs.index_put_({torch::indexing::Ellipsis, i}, 1);
    auto jac_row = torch::autograd::grad({out.tensor()}, {in.tensor()}, {grad_outputs}, true)[0];
    dout_din.tensor().base_index_put({i, torch::indexing::Slice()}, jac_row);
  }

  in.tensor().requires_grad_(was_ad);

  return {out, dout_din};
}
