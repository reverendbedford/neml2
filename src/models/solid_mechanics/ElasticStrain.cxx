#include "models/solid_mechanics/ElasticStrain.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
template <bool rate>
ElasticStrainTempl<rate>::ElasticStrainTempl(const std::string & name)
  : Model(name),
    total_strain(
        declareInputVariable<SymR2>("forces", rate ? "total_strain_rate" : "total_strain")),
    plastic_strain(
        declareInputVariable<SymR2>("state", rate ? "plastic_strain_rate" : "plastic_strain")),
    elastic_strain(
        declareOutputVariable<SymR2>("state", rate ? "elastic_strain_rate" : "elastic_strain"))
{
  setup();
}

template <bool rate>
void
ElasticStrainTempl<rate>::set_value(LabeledVector in,
                                    LabeledVector out,
                                    LabeledMatrix * dout_din) const
{
  // Simple additive decomposition:
  // elastic strain = total strain - plastic strain
  out.set(in.get<SymR2>(total_strain) - in.get<SymR2>(plastic_strain), elastic_strain);

  if (dout_din)
  {
    auto I = SymR2::identity_map().batch_expand(in.batch_size());
    dout_din->set(I, elastic_strain, total_strain);
    dout_din->set(-I, elastic_strain, plastic_strain);
  }
}

template class ElasticStrainTempl<true>;
template class ElasticStrainTempl<false>;
} // namespace neml2
