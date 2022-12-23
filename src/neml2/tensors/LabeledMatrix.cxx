#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/tensors/LabeledVector.h"

using namespace torch::indexing;

namespace neml2
{
LabeledMatrix::LabeledMatrix(const LabeledVector & A, const LabeledVector & B)
  : LabeledTensor<1, 2>(
        torch::zeros(
            utils::add_shapes(A.tensor().batch_sizes(), A.storage_size(), B.storage_size()),
            TorchDefaults),
        A.axis(0),
        B.axis(0))
{
  // Check that the two batch sizes were consistent
  if (A.batch_size() != B.batch_size())
    throw std::runtime_error("The batch sizes of the LabeledVectors are not consistent.");
}

LabeledMatrix::LabeledMatrix(const LabeledTensor<1, 2> & other)
  : LabeledTensor<1, 2>(other)
{
}

LabeledMatrix
LabeledMatrix::identity(TorchSize nbatch, const LabeledAxis & axis)
{
  return LabeledTensor<1, 2>(
      BatchTensor<1>::identity(axis.storage_size()).batch_expand_copy({nbatch}), axis, axis);
}

void
LabeledMatrix::accumulate(const LabeledMatrix & other, bool recursive)
{
  auto [idx0, idx0_other] = LabeledAxis::common_indices(axis(0), other.axis(0), recursive);
  auto [idx1, idx1_other] = LabeledAxis::common_indices(axis(1), other.axis(1), recursive);

  // This is annoying -- since we are using advanced indexing, torch actually creates a copy of the
  // indexed view (and so it's not a view anymore).
  auto temp = _tensor.base_index({idx0, Slice()});
  auto temp_other = other.tensor().base_index({idx0_other, Slice()});
  temp.base_index_put({Slice(), idx1},
                      temp.base_index({Slice(), idx1}) +
                          temp_other.base_index({Slice(), idx1_other}));

  _tensor.base_index_put({idx0, Slice()}, temp);
}

void
LabeledMatrix::fill(const LabeledMatrix & other, bool recursive)
{
  auto [idx0, idx0_other] = LabeledAxis::common_indices(axis(0), other.axis(0), recursive);
  auto [idx1, idx1_other] = LabeledAxis::common_indices(axis(1), other.axis(1), recursive);

  // This is annoying -- since we are using advanced indexing, torch actually creates a copy of the
  // indexed view (and so it's not a view anymore).
  auto temp = _tensor.base_index({idx0, Slice()});
  auto temp_other = other.tensor().base_index({idx0_other, Slice()});
  temp.base_index_put({Slice(), idx1}, temp_other.base_index({Slice(), idx1_other}));

  _tensor.base_index_put({idx0, Slice()}, temp);
}

LabeledMatrix
LabeledMatrix::chain(const LabeledMatrix & other) const
{
  // This function expresses a chain rule, which is just a dot product between the values of this
  // and the values of the input The main annoyance is just getting the names correct

  // Check that we are conformal
  if (batch_size() != other.batch_size())
    throw std::runtime_error("LabeledMatrix batch sizes are "
                             "not the same");
  if (axis(1) != other.axis(0))
    throw std::runtime_error("Label objects are not conformal");

  // If all the sizes are correct then executing the chain rule is pretty easy
  return LabeledMatrix(torch::bmm(tensor(), other.tensor()), axis(0), other.axis(1));
}

LabeledMatrix
LabeledMatrix::inverse() const
{
  // Make debug
  if (axis(0).storage_size() != axis(1).storage_size())
    throw std::runtime_error("Can only invert square derivatives");

  return LabeledMatrix(torch::linalg::inv(tensor()), axis(1), axis(0));
}

void
LabeledMatrix::write(std::ostream & os, std::string delimiter, TorchSize batch, bool header) const
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

  for (auto row_name : axis(1).item_names())
  {
    os << row_name;
    for (auto col_name : axis(0).item_names())
    {
      TorchSize sz = axis(0).storage_size(col_name);
      if (sz == 0)
        continue;
      else
        for (TorchSize i = 0; i < sz; i++)
          os << delimiter << (*this)(col_name, row_name).index({batch, i}).item<double>();
    }
    os << std::endl;
  }
  os << std::endl;
}
} // namespace neml2
