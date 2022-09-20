#include "SmallStrainIsotropicLinearElasticModel.h"

SmallStrainIsotropicLinearElasticModel::
SmallStrainIsotropicLinearElasticModel(const Scalar & E, 
                                       const Scalar & nu) :
    _E(E), _nu(nu)
{

}

void
SmallStrainIsotropicLinearElasticModel::update(
    State & state_np1, const State & forces_np1, 
    const State & state_n, const State & forces_n)
{
  
}

StateInfo 
SmallStrainIsotropicLinearElasticModel::internal_state() const 
{
  return StateInfo();
}
