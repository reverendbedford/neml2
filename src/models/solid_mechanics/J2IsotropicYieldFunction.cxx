#include "models/solid_mechanics/J2IsotropicYieldFunction.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
J2IsotropicYieldFunction::J2IsotropicYieldFunction(const std::string & name)
  : YieldFunction(name),
    _g_idx(declareVariable<Scalar>(input(), "state", "isotropic_hardening"))
{
  setup();
}

void
J2IsotropicYieldFunction::set_value(LabeledVector in,
                                    LabeledVector out,
                                    LabeledMatrix * dout_din) const
{
  // First retrieve the hardening variables
  auto mandel = in.get<SymR2>(_mandel_idx);
  auto g = in.get<Scalar>(_g_idx);

  // Compute the yield function
  auto s = mandel.dev();
  auto J2 = s.norm();
  auto H = sqrt(2.0 / 3.0) * g;
  auto f = J2 - H;

  // Set the output
  out.set(f, _f_idx);

  if (dout_din)
  {
    // Compute the yield function derivative
    auto df_dmandel = s / (s.norm() + EPS);
    auto df_dg = Scalar(-sqrt(2.0 / 3.0), in.batch_size());

    // Set the output
    dout_din->set(df_dmandel, _f_idx, _mandel_idx);
    dout_din->set(df_dg, _f_idx, _g_idx);
  }
}

void
J2IsotropicYieldFunction::set_dvalue(LabeledVector in,
                                     LabeledMatrix dout_din,
                                     LabeledTensor<1, 3> * d2out_din2) const
{
  // First retrieve the hardening variables
  auto mandel = in.get<SymR2>(_mandel_idx);
  auto g = in.get<Scalar>(_g_idx);

  // Compute the yield function
  auto s = mandel.dev();

  // Compute the yield function derivative
  auto df_dmandel = s / (s.norm() + EPS);
  auto df_dg = Scalar(-sqrt(2.0 / 3.0), in.batch_size());

  // Set the output
  dout_din.set(df_dmandel, _f_idx, _mandel_idx);
  dout_din.set(df_dg, _f_idx, _g_idx);

  if (d2out_din2)
  {
    // Compute the yield function second derivative
    auto n = s / (s.norm() + EPS);
    auto I = SymSymR4::init(SymSymR4::FillMethod::identity_sym);
    auto J = SymSymR4::init(SymSymR4::FillMethod::identity_dev);
    auto d2f_dmandel2 = (I - n.outer(n)) * J / (s.norm() + EPS);

    // Set the output
    d2out_din2->set(d2f_dmandel2, _f_idx, _mandel_idx, _mandel_idx);
  }
}
} // namespace neml2
