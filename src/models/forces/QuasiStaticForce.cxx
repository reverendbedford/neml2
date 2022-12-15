#include "models/forces/QuasiStaticForce.h"

namespace neml2
{
template <typename T, bool stateful>
QuasiStaticForce<T, stateful>::QuasiStaticForce(const std::string & name)
  : Force<T, stateful>(name)
{
  this->input().subaxis("forces").template add<T>(name);
  this->output().subaxis("forces").template add<T>(name);

  if constexpr (stateful)
  {
    this->input().subaxis("old_forces").template add<T>(name);
    this->output().subaxis("old_forces").template add<T>(name);
  }

  this->setup();
}

template <typename T, bool stateful>
void
QuasiStaticForce<T, stateful>::set_value(LabeledVector in,
                                         LabeledVector out,
                                         LabeledMatrix * dout_din) const
{
  out.slice("forces").set(in.slice("forces")(this->name()), this->name());

  if constexpr (stateful)
    out.slice("old_forces").set(in.slice("old_forces")(this->name()), this->name());

  if (dout_din)
  {
    auto I = T::identity_map().batch_expand(in.batch_size());
    dout_din->block("forces", "forces").set(I, this->name(), this->name());

    if constexpr (stateful)
      dout_din->block("old_forces", "old_forces").set(I, this->name(), this->name());
  }
}

template class QuasiStaticForce<Scalar, true>;
template class QuasiStaticForce<Scalar, false>;
template class QuasiStaticForce<SymR2, true>;
template class QuasiStaticForce<SymR2, false>;
} // namespace neml2
