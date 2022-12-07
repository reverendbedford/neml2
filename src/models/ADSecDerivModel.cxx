#include "models/ADSecDerivModel.h"

std::tuple<LabeledVector, LabeledMatrix>
ADSecDerivModel::value_and_dvalue(LabeledVector in) const
{
  bool was_ad = true;
  if (!in.tensor().requires_grad())
  {
    was_ad = false;
    in.tensor().requires_grad_();
  }

  // Evalute the model (not its derivatives)
  auto out = SecDerivModel::value(in);

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

std::tuple<LabeledMatrix, LabeledTensor<1, 3>>
ADSecDerivModel::dvalue_and_d2value(LabeledVector in) const
{
  bool was_ad = true;
  if (!in.tensor().requires_grad())
  {
    was_ad = false;
    in.tensor().requires_grad_();
  }

  // Evalute the model's first derivative (not its second derivatives)
  auto dout_din = SecDerivModel::dvalue(in);

  // Allocate space for Jacobian
  LabeledTensor<1, 3> d2out_din2(in.batch_size(), output(), in.axis(0), in.axis(0));

  // Loop over rows of the state to retrieve the derivatives
  for (TorchSize i = 0; i < dout_din.tensor().base_sizes()[0]; i++)
    for (TorchSize j = 0; j < dout_din.tensor().base_sizes()[1]; j++)
    {
      BatchTensor<1> grad_outputs = torch::zeros_like(dout_din.tensor());
      grad_outputs.index_put_({torch::indexing::Ellipsis, i, j}, 1);
      auto jac_row =
          torch::autograd::grad({dout_din.tensor()}, {in.tensor()}, {grad_outputs}, true)[0];
      d2out_din2.tensor().base_index_put({i, j, torch::indexing::Slice()}, jac_row);
    }

  in.tensor().requires_grad_(was_ad);

  return {dout_din, d2out_din2};
}
