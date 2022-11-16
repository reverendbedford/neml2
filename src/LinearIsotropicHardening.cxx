#include "LinearIsotropicHardening.h"
#include "SymSymR4.h"

LinearIsotropicHardening::LinearIsotropicHardening(Scalar s0, Scalar K)
  : _s0(s0),
    _K(K)
{
}

State
LinearIsotropicHardening::value(State input)
{
  State res = State::same_batch(output(), input);
  res.set<SymR2>("stress", input.get<SymR2>("stress"));
  res.set<Scalar>("isotropic_hardening", _s0 + _K * input.get<Scalar>("equivalent_plastic_strain"));
  return res;
}

StateDerivative
LinearIsotropicHardening::dvalue(State input)
{
  StateDerivative res = StateDerivative::same_batch(output(), input);
  res.set<SymSymR4>(
      "stress",
      "stress",
      SymSymR4::init(SymSymR4::FillMethod::identity_sym).expand_batch(input.batch_size()));
  res.set<Scalar>(
      "isotropic_hardening", "equivalent_plastic_strain", _K.expand_batch(input.batch_size()));
  return res;
}
