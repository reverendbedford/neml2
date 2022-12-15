#include "models/solid_mechanics/TotalStrain.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
template <bool rate>
TotalStrainTempl<rate>::TotalStrainTempl(const std::string & name)
  : Model(name),
    elastic_strain(
        declareInputVariable<SymR2>({"state", rate ? "elastic_strain_rate" : "elastic_strain"})),
    plastic_strain(
        declareInputVariable<SymR2>({"state", rate ? "plastic_strain_rate" : "plastic_strain"})),
    total_strain(
        declareOutputVariable<SymR2>({"state", rate ? "total_strain_rate" : "total_strain"}))
{
  this->setup();
}

template <bool rate>
void
TotalStrainTempl<rate>::set_value(LabeledVector in,
                                  LabeledVector out,
                                  LabeledMatrix * dout_din) const
{
  // total strain = elastic strain + plastic strain
  out.set(in.get<SymR2>(elastic_strain) + in.get<SymR2>(plastic_strain), total_strain);

  if (dout_din)
  {
    auto I = SymR2::identity_map().batch_expand(in.batch_size());
    dout_din->set(I, total_strain, elastic_strain);
    dout_din->set(I, total_strain, plastic_strain);
  }
}

template class TotalStrainTempl<true>;
template class TotalStrainTempl<false>;
} // namespace neml2
