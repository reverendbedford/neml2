#include "models/solid_mechanics/LinearIsotropicElasticity.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
template <bool rate>
LinearIsotropicElasticityTempl<rate>::LinearIsotropicElasticityTempl(const std::string & name,
                                                                     Scalar E,
                                                                     Scalar nu)
  : Model(name),
    _E(register_parameter("youngs_modulus", E)),
    _nu(register_parameter("poissons_ratio", nu))
{
  input().add<LabeledAxis>("state");
  input().subaxis("state").add<SymR2>(rate ? "elastic_strain_rate" : "elastic_strain");

  output().add<LabeledAxis>("state");
  output().subaxis("state").add<SymR2>(rate ? "cauchy_stress_rate" : "cauchy_stress");

  setup();
}

template <bool rate>
void
LinearIsotropicElasticityTempl<rate>::set_value(LabeledVector in,
                                                LabeledVector out,
                                                LabeledMatrix * dout_din) const
{
  // Retrieve whatever we need from the input,
  // Here we need the elastic strain
  auto Ee = in.slice(0, "state").get<SymR2>(rate ? "elastic_strain_rate" : "elastic_strain");

  // Compute the cauchy stress
  auto C = SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {_E, _nu});
  auto cauchy = C * Ee;

  // Set the output
  out.slice(0, "state").set(cauchy, rate ? "cauchy_stress_rate" : "cauchy_stress");

  if (dout_din)
  {
    // Set the output
    dout_din->block("state", "state")
        .set(C.expand_batch(in.batch_size()),
             rate ? "cauchy_stress_rate" : "cauchy_stress",
             rate ? "elastic_strain_rate" : "elastic_strain");
  }
}

template class LinearIsotropicElasticityTempl<true>;
template class LinearIsotropicElasticityTempl<false>;
} // namespace neml2
