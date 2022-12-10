#include "models/SecDerivModel.h"

namespace neml2
{
LabeledMatrix
SecDerivModel::dvalue(LabeledVector in) const
{
  LabeledMatrix dout_din(in.batch_size(), output(), in.axis(0));
  set_dvalue(in, dout_din);
  return dout_din;
}

LabeledTensor<1, 3>
SecDerivModel::d2value(LabeledVector in) const
{
  auto [dout_din, d2out_din2] = dvalue_and_d2value(in);
  return d2out_din2;
}

std::tuple<LabeledMatrix, LabeledTensor<1, 3>>
SecDerivModel::dvalue_and_d2value(LabeledVector in) const
{
  LabeledMatrix dout_din(in.batch_size(), output(), in.axis(0));
  LabeledTensor<1, 3> d2out_din2(in.batch_size(), output(), in.axis(0), in.axis(0));
  set_dvalue(in, dout_din, &d2out_din2);
  return {dout_din, d2out_din2};
}
} // namespace neml2
