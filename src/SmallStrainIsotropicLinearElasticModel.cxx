#include "SmallStrainIsotropicLinearElasticModel.h"
#include "ElasticityTensors.h"
#include "BatchedSymSymR4.h"

SmallStrainIsotropicLinearElasticModel::SmallStrainIsotropicLinearElasticModel(const Scalar & E,
                                                                               const Scalar & nu)
  : _E(E),
    _nu(nu),
    _C(fill_isotropic(E, nu))
{
}

void
SmallStrainIsotropicLinearElasticModel::update(State & state_np1,
                                               const State & forces_np1,
                                               const State & state_n,
                                               const State & forces_n)
{
  // stress_np1 = C : strain_np1...
  state_np1.set<BatchedSymR2>("stress", _C.dot(forces_np1.get<BatchedSymR2>("strain")));
}

void
SmallStrainIsotropicLinearElasticModel::update_linearized(StateDerivative & tangent,
                                                          const State & forces_np1,
                                                          const State & state_n,
                                                          const State & forces_n)
{
  // Just repeat the elasticity tensor along the batch dimension
  tangent.set<BatchedSymSymR4>(
      "stress",
      "strain",
      torch::repeat_interleave(_C.reshape({1, 6, 6}), forces_np1.batch_size(), 0));
}

StateInfo
SmallStrainIsotropicLinearElasticModel::internal_state() const
{
  return StateInfo();
}
