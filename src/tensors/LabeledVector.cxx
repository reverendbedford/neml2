#include "tensors/LabeledVector.h"
#include "tensors/LabeledMatrix.h"

namespace neml2
{
LabeledVector::LabeledVector(const LabeledTensor<1, 1> & other)
  : LabeledTensor<1, 1>(other)
{
}

void
LabeledVector::accumulate(const LabeledVector & other, bool recursive)
{
  auto [idx, idx_other] = LabeledAxis::common_indices(axis(0), other.axis(0), recursive);
  _tensor.base_index_put({idx}, _tensor.base_index({idx}) + other.tensor().base_index({idx_other}));
}

void
LabeledVector::fill(const LabeledVector & other, bool recursive)
{
  auto [idx, idx_other] = LabeledAxis::common_indices(axis(0), other.axis(0), recursive);
  _tensor.base_index_put({idx}, other.tensor().base_index({idx_other}));
}

LabeledMatrix
LabeledVector::outer(const LabeledVector & other) const
{
  return LabeledMatrix(
      torch::bmm(tensor().unsqueeze(2), other.tensor().unsqueeze(1)), axis(0), other.axis(0));
}

void
LabeledVector::write(std::ostream & os, std::string delimiter, TorchSize batch, bool header) const
{
  if (header)
  {
    for (auto name : axis(0).item_names())
    {
      TorchSize sz = axis(0).storage_size(name);
      if (sz == 0)
        continue;
      else if (sz == 1)
        os << delimiter << name;
      else
        for (TorchSize i = 0; i < sz; i++)
          os << delimiter << name << "_" << i;
    }
    os << std::endl;
  }

  for (auto name : axis(0).item_names())
  {
    TorchSize sz = axis(0).storage_size(name);
    if (sz == 0)
      continue;
    else
      for (TorchSize i = 0; i < sz; i++)
        os << delimiter << (*this)(name).index({batch, i}).item<double>();
  }
  os << std::endl;
}
} // namespace neml2
