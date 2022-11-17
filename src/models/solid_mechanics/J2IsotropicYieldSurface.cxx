#include "models/solid_mechanics/J2IsotropicYieldSurface.h"

StateInfo
J2IsotropicYieldSurface::hardening_interface() const
{
  StateInfo interface;
  interface.add<Scalar>("isotropic_hardening");
  return interface;
}

Scalar
J2IsotropicYieldSurface::f(const State & interface) const
{
  auto J2 = sqrt(3.0 / 2.0) * interface.get<SymR2>("stress").dev().norm();
  auto H = sqrt(2.0 / 3.0) * interface.get<Scalar>("isotropic_hardening");
  return J2 - H;
}

State
J2IsotropicYieldSurface::df_ds(const State & interface) const
{
  // We have the usual problem with zero stress
  auto res = interface.clone();

  auto s = interface.get<SymR2>("stress").dev();
  res.set<SymR2>("stress", sqrt(3.0 / 2.0) * s / (s.norm() + EPS));
  res.set<Scalar>("isotropic_hardening", Scalar(-sqrt(2.0 / 3.0), interface.batch_size()));

  return res;
}

StateDerivative
J2IsotropicYieldSurface::d2f_ds2(const State & interface) const
{
  auto res = StateDerivative(interface, interface);
  SymR2 s = interface.get<SymR2>("stress").dev();

  SymR2 n = s / (s.norm() + EPS);
  SymSymR4 I = SymSymR4::init(SymSymR4::FillMethod::identity_sym);
  SymSymR4 J = SymSymR4::init(SymSymR4::FillMethod::identity_dev);
  SymSymR4 nn = I - n.outer(n);
  SymSymR4 r = nn * J;

  res.set<SymSymR4>("stress", "stress", r * sqrt(3.0 / 2.0) / (s.norm() + EPS).unsqueeze(-1));

  return res;
}
