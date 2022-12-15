#include "models/IdentityMap.h"
#include "tensors/SymSymR4.h"

namespace neml2
{
template <typename T>
IdentityMap<T>::IdentityMap(const std::string & name,
                            const std::vector<std::string> & from_var,
                            const std::vector<std::string> & to_var)
  : Model(name),
    from(declareInputVariable<T>(from_var)),
    to(declareOutputVariable<T>(to_var))
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
