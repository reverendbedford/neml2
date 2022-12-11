#include "models/solid_mechanics/J2IsotropicYieldFunction.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
J2IsotropicYieldFunction::J2IsotropicYieldFunction(const std::string & name)
  : YieldFunction(name)
{
  input().subaxis("state").add<Scalar>("isotropic_hardening");
  setup();
}

void
J2IsotropicYieldFunction::set_value(LabeledVector in,
                                    LabeledVector out,
                                    LabeledMatrix * dout_din) const
{
  // First retrieve the hardening variables
  auto mandel = in.slice(0, "state").get<SymR2>("mandel_stress");
  auto h = in.slice(0, "state").get<Scalar>("isotropic_hardening");

  // Compute the yield function
  auto s = mandel.dev();
  auto J2 = s.norm();
  auto H = sqrt(2.0 / 3.0) * h;
  auto f = J2 - H;

  // Set the output
  out.slice(0, "state").set(f, "yield_function");

  if (dout_din)
  {
    // Compute the yield function derivative
    auto df_dmandel = s / (s.norm() + EPS);
    auto df_dh = Scalar(-sqrt(2.0 / 3.0), in.batch_size());

    // Set the output
    dout_din->block("state", "state").set(df_dmandel, "yield_function", "mandel_stress");
    dout_din->block("state", "state").set(df_dh, "yield_function", "isotropic_hardening");
  }
}

void
J2IsotropicYieldFunction::set_dvalue(LabeledVector in,
                                     LabeledMatrix dout_din,
                                     LabeledTensor<1, 3> * d2out_din2) const
{
  // First retrieve the hardening variables
  auto mandel = in.slice(0, "state").get<SymR2>("mandel_stress");
  auto h = in.slice(0, "state").get<Scalar>("isotropic_hardening");

  // Compute the yield function
  auto s = mandel.dev();

  // Compute the yield function derivative
  auto df_dmandel = s / (s.norm() + EPS);
  auto df_dh = Scalar(-sqrt(2.0 / 3.0), in.batch_size());

  // Set the output
  dout_din.block("state", "state").set(df_dmandel, "yield_function", "mandel_stress");
  dout_din.block("state", "state").set(df_dh, "yield_function", "isotropic_hardening");

  if (d2out_din2)
  {
    // Compute the yield function second derivative
    auto n = s / (s.norm() + EPS);
    auto I = SymSymR4::init(SymSymR4::FillMethod::identity_sym);
    auto J = SymSymR4::init(SymSymR4::FillMethod::identity_dev);
    auto d2f_dmandel2 = (I - n.outer(n)) * J / (s.norm() + EPS);

    // Set the output
    d2out_din2->block("state", "state", "state")
        .set(d2f_dmandel2, "yield_function", "mandel_stress", "mandel_stress");
  }
}
} // namespace neml2
