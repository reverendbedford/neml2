#include "neml2/models/SumModel.h"
#include "neml2/tensors/SymSymR4.h"

namespace neml2
{
template <typename T>
SumModel<T>::SumModel(const std::string & name,
                      const std::vector<std::vector<std::string>> & from_var,
                      const std::vector<std::string> & to_var)
  : Model(name),
    to(declareOutputVariable<T>(to_var))
{
  for (auto fv : from_var)
    from.push_back(declareInputVariable<T>(fv));

  this->setup();
}

template <typename T>
void
SumModel<T>::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  auto sum = T::zeros(in.batch_size());
  for (auto fv : from)
    sum += in(fv);
  out.set(sum, to);

  if (dout_din)
  {
    for (auto fv : from)
      dout_din->set(T::identity_map().batch_expand(in.batch_size()), to, fv);
  }
}

template class SumModel<Scalar>;
template class SumModel<SymR2>;
} // namespace neml2
