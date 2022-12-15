#include "models/solid_mechanics/ElasticStrain.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
template <bool rate>
ElasticStrainTempl<rate>::ElasticStrainTempl(const std::string & name)
  : Model(name)
{
  this->input().template add<LabeledAxis>("forces");
  this->input().subaxis("forces").template add<SymR2>(rate ? "total_strain_rate" : "total_strain");

  this->input().template add<LabeledAxis>("state");
  this->input().subaxis("state").template add<SymR2>(rate ? "plastic_strain_rate"
                                                          : "plastic_strain");

  this->output().template add<LabeledAxis>("state");
  this->output().subaxis("state").template add<SymR2>(rate ? "elastic_strain_rate"
                                                           : "elastic_strain");

  this->setup();
}

template <bool rate>
void
ElasticStrainTempl<rate>::set_value(LabeledVector in,
                                    LabeledVector out,
                                    LabeledMatrix * dout_din) const
{
  // Retrieve whatever we need from the input,
  // Here we need the total strain
  auto total_strain = in.slice("forces").get<SymR2>(rate ? "total_strain_rate" : "total_strain");
  auto plastic_strain =
      in.slice("state").get<SymR2>(rate ? "plastic_strain_rate" : "plastic_strain");

  auto elastic_strain = total_strain - plastic_strain;

  // Set the output
  out.slice("state").set(elastic_strain, rate ? "elastic_strain_rate" : "elastic_strain");

  if (dout_din)
  {
    auto I = SymR2::identity_map().batch_expand(in.batch_size());

    // Set the output
    dout_din->block("state", "forces")
        .set(I,
             rate ? "elastic_strain_rate" : "elastic_strain",
             rate ? "total_strain_rate" : "total_strain");
    dout_din->block("state", "state")
        .set(-I,
             rate ? "elastic_strain_rate" : "elastic_strain",
             rate ? "plastic_strain_rate" : "plastic_strain");
  }
}

template class ElasticStrainTempl<true>;
template class ElasticStrainTempl<false>;
} // namespace neml2
