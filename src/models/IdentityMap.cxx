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
    _from_axis_name(from_axis_name),
    _from_var_name(from_var_name),
    _to_axis_name(to_axis_name),
    _to_var_name(to_var_name)
{
  this->input().template add<LabeledAxis>(from_axis_name);
  this->input().subaxis(from_axis_name).template add<T>(from_var_name);

  this->output().template add<LabeledAxis>(to_axis_name);
  this->output().subaxis(to_axis_name).template add<T>(to_var_name);

  this->setup();
}

template <typename T>
void
IdentityMap<T>::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  out.slice(0, _to_axis_name).set(in.slice(0, _from_axis_name)(_from_var_name), _to_var_name);
  if (dout_din)
  {
    auto I = T::identity_map().batch_expand(in.batch_size());
    dout_din->block(_to_axis_name, _from_axis_name).set(I, _to_var_name, _from_var_name);
  }
}

template class IdentityMap<Scalar>;
template class IdentityMap<SymR2>;
} // namespace neml2
