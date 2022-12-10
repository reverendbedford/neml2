#include "SampleSecDerivModel.h"

using namespace neml2;

template <bool is_ad>
SampleSecDerivModelTempl<is_ad>::SampleSecDerivModelTempl(const std::string & name)
  : SampleSecDerivModelBase<is_ad>(name)
{
  this->input().template add<LabeledAxis>("state");
  this->input().subaxis("state").template add<Scalar>("x1");
  this->input().subaxis("state").template add<Scalar>("x2");

  this->output().template add<LabeledAxis>("state");
  this->output().subaxis("state").template add<Scalar>("y");

  this->setup();
}

template <bool is_ad>
void
SampleSecDerivModelTempl<is_ad>::set_value(LabeledVector in,
                                           LabeledVector out,
                                           LabeledMatrix * dout_din) const
{
  // Grab the inputs
  auto x1 = in.slice(0, "state").get<Scalar>("x1");
  auto x2 = in.slice(0, "state").get<Scalar>("x2");

  // y = x1^3 + x2^4
  auto y = x1 * x1 * x1 + x2 * x2 * x2 * x2;

  // Set the output
  out.slice(0, "state").set(y, "y");

  if constexpr (!is_ad)
    if (dout_din)
    {
      auto dy_dx1 = 3 * x1 * x1;
      auto dy_dx2 = 4 * x2 * x2 * x2;

      dout_din->block("state", "state").set(dy_dx1, "y", "x1");
      dout_din->block("state", "state").set(dy_dx2, "y", "x2");
    }
}

template <bool is_ad>
void
SampleSecDerivModelTempl<is_ad>::set_dvalue(LabeledVector in,
                                            LabeledMatrix dout_din,
                                            LabeledTensor<1, 3> * d2out_din2) const
{
  // Grab the inputs
  auto x1 = in.slice(0, "state").get<Scalar>("x1");
  auto x2 = in.slice(0, "state").get<Scalar>("x2");

  // y = x1^3 + x2^4
  auto dy_dx1 = 3 * x1 * x1;
  auto dy_dx2 = 4 * x2 * x2 * x2;

  // Set the output
  dout_din.block("state", "state").set(dy_dx1, "y", "x1");
  dout_din.block("state", "state").set(dy_dx2, "y", "x2");

  if constexpr (!is_ad)
    if (d2out_din2)
    {
      auto d2y_dx12 = 6 * x1;
      auto d2y_dx22 = 12 * x2 * x2;

      d2out_din2->block("state", "state", "state").set(d2y_dx12, "y", "x1", "x1");
      d2out_din2->block("state", "state", "state").set(d2y_dx22, "y", "x2", "x2");
    }
}

template class SampleSecDerivModelTempl<true>;
template class SampleSecDerivModelTempl<false>;
