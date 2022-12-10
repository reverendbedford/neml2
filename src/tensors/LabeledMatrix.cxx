#include "tensors/LabeledMatrix.h"
#include "tensors/LabeledVector.h"

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

void
LabeledMatrix::assemble(const LabeledMatrix & other)
{
  // First fill in all the variables
  for (auto var_itr_i : other.axis(0).variables())
    if (axis(0).has_variable(var_itr_i.first))
      for (auto var_itr_j : other.axis(1).variables())
        if (axis(1).has_variable(var_itr_j.first))
          (*this)(var_itr_i.first, var_itr_j.first) += other(var_itr_i.first, var_itr_j.first);

  // Then recursively fill the subaxis
  for (auto subaxis_itr_i : other.axis(0).subaxes())
    if (axis(0).has_subaxis(subaxis_itr_i.first))
      for (auto subaxis_itr_j : other.axis(1).subaxes())
        if (axis(1).has_subaxis(subaxis_itr_j.first))
          LabeledMatrix(block(subaxis_itr_i.first, subaxis_itr_j.first))
              .assemble(other.block(subaxis_itr_i.first, subaxis_itr_j.first));
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
