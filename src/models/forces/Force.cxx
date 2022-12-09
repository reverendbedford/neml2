#include "models/forces/Force.h"

namespace neml2
{
template <typename T, bool stateful>
Force<T, stateful>::Force(const std::string & name)
  : Model(name)
{
  this->input().add<LabeledAxis>("forces");
  this->output().add<LabeledAxis>("forces");

  if constexpr (stateful)
  {
    this->input().add<LabeledAxis>("old_forces");
    this->output().add<LabeledAxis>("old_forces");
  }

  this->setup();
}

template class Force<Scalar, true>;
template class Force<Scalar, false>;
template class Force<SymR2, true>;
template class Force<SymR2, false>;
} // namespace neml2
