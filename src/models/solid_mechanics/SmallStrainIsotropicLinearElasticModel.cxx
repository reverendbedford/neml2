#include "models/solid_mechanics/SmallStrainIsotropicLinearElasticModel.h"
#include "tensors/SymSymR4.h"

SmallStrainIsotropicLinearElasticModel::SmallStrainIsotropicLinearElasticModel(const Scalar & E,
                                                                               const Scalar & nu)
  : _E(E),
    _nu(nu),
    _C(SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {E, nu}))
{
}

State
SmallStrainIsotropicLinearElasticModel::value(StateInput input)
{
  State forces_np1 = input[0];
  State state_n = input[1];
  State forces_n = input[2];

  State res = State(output(), forces_np1.batch_size());
  res.set<SymR2>("stress", _C * forces_np1.get<SymR2>("strain"));

  return res;
}

StateDerivativeOutput
SmallStrainIsotropicLinearElasticModel::dvalue(StateInput input)
{
  State forces_np1 = input[0];
  State state_n = input[1];
  State forces_n = input[2];

  StateDerivative dF_np1 = StateDerivative(output(), forces_np1.info(), forces_np1.batch_size());
  StateDerivative dF_n = StateDerivative(output(), forces_n.info(), forces_n.batch_size());
  StateDerivative dS_n = StateDerivative(output(), state_n.info(), state_n.batch_size());

  dF_np1.set<SymSymR4>("stress", "strain", _C.expand_batch(forces_np1.batch_size()));

  return {dF_np1, dS_n, dF_n};
}
