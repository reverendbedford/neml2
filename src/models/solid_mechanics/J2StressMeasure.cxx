#include "models/solid_mechanics/J2StressMeasure.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
J2StressMeasure::J2StressMeasure(const std::string & name)
  : StressMeasure(name)
{
  setup();
}

void
J2StressMeasure::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  // First retrieve the over stress
  auto stress = in.get<SymR2>(overstress);

  // Set the output
  auto s = stress.dev();
  auto J2 = s.norm();
  out.set(J2, stress_measure);

  if (dout_din)
  {
    // Compute the yield function derivative
    auto df_dstress = s / (J2 + EPS);

    // Set the output
    dout_din->set(df_dstress, stress_measure, overstress);
  }
}

void
J2StressMeasure::set_dvalue(LabeledVector in,
                            LabeledMatrix dout_din,
                            LabeledTensor<1, 3> * d2out_din2) const
{
  // First retrieve the Mandel stress
  auto stress = in.get<SymR2>(overstress);

  // Calculate current value...
  auto s = stress.dev();
  auto J2 = s.norm();

  // Compute the yield function derivative
  auto df_dm = s / (J2 + EPS);

  // Set the output
  dout_din.set(df_dm, stress_measure, overstress);

  if (d2out_din2)
  {
    // Compute the yield function second derivative
    auto I = SymSymR4::init(SymSymR4::FillMethod::identity_sym);
    auto J = SymSymR4::init(SymSymR4::FillMethod::identity_dev);
    auto d2f_dstress2 = (I - df_dm.outer(df_dm)) * J / (J2 + EPS);

    // Set the output
    d2out_din2->set(d2f_dstress2, stress_measure, overstress, overstress);
  }
}

}
