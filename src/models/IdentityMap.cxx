#include "models/IdentityMap.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
template <typename T>
IdentityMap<T>::IdentityMap(const std::string & name,
                            const std::string & from_axis_name,
                            const std::string & from_var_name,
                            const std::string & to_axis_name,
                            const std::string & to_var_name)
  : Model(name),
    from(declareInputVariable<T>(from_axis_name, from_var_name)),
    to(declareOutputVariable<T>(to_axis_name, to_var_name))
{
  this->setup();
}

template <typename T>
void
IdentityMap<T>::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  out.set(in(from), to);
  if (dout_din)
  {
    auto I = T::identity_map().batch_expand(in.batch_size());
    dout_din->set(I, to, from);
  }
}

template class IdentityMap<Scalar>;
template class IdentityMap<SymR2>;
} // namespace neml2
