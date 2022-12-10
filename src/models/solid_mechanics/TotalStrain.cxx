#include "models/solid_mechanics/TotalStrain.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
template <bool rate>
TotalStrainTempl<rate>::TotalStrainTempl(const std::string & name)
  : Model(name)
{
  this->input().template add<LabeledAxis>("state");
  this->input().subaxis("state").template add<SymR2>(rate ? "plastic_strain_rate"
                                                          : "plastic_strain");
  this->input().subaxis("state").template add<SymR2>(rate ? "elastic_strain_rate"
                                                          : "elastic_strain");

  this->output().template add<LabeledAxis>("state");
  this->output().subaxis("state").template add<SymR2>(rate ? "total_strain_rate" : "total_strain");

  this->setup();
}

template <bool rate>
void
TotalStrainTempl<rate>::set_value(LabeledVector in,
                                  LabeledVector out,
                                  LabeledMatrix * dout_din) const
{
  // Retrieve whatever we need from the input,
  // Here we need the elastic/plastic strains
  auto elastic_strain =
      in.slice(0, "state").get<SymR2>(rate ? "elastic_strain_rate" : "elastic_strain");
  auto plastic_strain =
      in.slice(0, "state").get<SymR2>(rate ? "plastic_strain_rate" : "plastic_strain");

  auto total_strain = elastic_strain + plastic_strain;

  // Set the output
  out.slice(0, "state").set(total_strain, rate ? "total_strain_rate" : "total_strain");

  if (dout_din)
  {
    auto I = SymR2::identity_map().batch_expand(in.batch_size());

    // Set the output
    dout_din->block("state", "forces")
        .set(I,
             rate ? "total_strain_rate" : "total_strain",
             rate ? "elastic_strain_rate" : "elastic_strain");
    dout_din->block("state", "state")
        .set(I,
             rate ? "total_strain_rate" : "total_strain",
             rate ? "plastic_strain_rate" : "plastic_strain");
  }
}

template class TotalStrainTempl<true>;
template class TotalStrainTempl<false>;
} // namespace neml2
