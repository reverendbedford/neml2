#include "models/solid_mechanics/LinearIsotropicElasticity.h"
#include "tensors/SymSymR4.h"

template <bool rate>
LinearIsotropicElasticity<rate>::LinearIsotropicElasticity(const std::string & name,
                                                           Scalar E,
                                                           Scalar nu)
  : Model(name),
    _E(E),
    _nu(nu)
{
  input().add<LabeledAxis>("state");
  input().subaxis("state").add<SymR2>(rate ? "elastic_strain_rate" : "elastic_strain");

  output().add<LabeledAxis>("state");
  output().subaxis("state").add<SymR2>(rate ? "cauchy_stress_rate" : "cauchy_stress");

  setup();
}

template <bool rate>
void
LinearIsotropicElasticity<rate>::set_value(LabeledVector in,
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

template class LinearIsotropicElasticity<true>;
template class LinearIsotropicElasticity<false>;
