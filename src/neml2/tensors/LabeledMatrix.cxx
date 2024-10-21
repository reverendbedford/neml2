// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/tensors/LabeledVector.h"
#include "neml2/misc/math.h"

namespace neml2
{
LabeledMatrix
LabeledMatrix::identity(TensorShapeRef batch_size,
                        const LabeledAxis & axis,
                        const torch::TensorOptions & options)
{
  return LabeledMatrix(Tensor::identity(batch_size, axis.storage_size(), options), {&axis, &axis});
}

void
LabeledMatrix::fill(const LabeledMatrix & other, bool recursive)
{
  neml_assert_dbg(axis(1) == other.axis(1), "Can only accumulate matrices with conformal y axes");
  const auto indices = axis(0).common_indices(other.axis(0), recursive);
  for (const auto & [idxi, idxi_other] : indices)
    _tensor.base_index_put_({idxi}, other.tensor().base_index({idxi_other}));
}

std::map<LabeledAxisAccessor, Tensor>
LabeledMatrix::split_variables(Size i, bool qualified) const
{
  auto vars = axis(i).variable_names();
  std::vector<Size> split_size;
  for (const auto & var : vars)
    split_size.push_back(axis(i).storage_size(var));

  auto vals = tensor().split(split_size, batch_dim() + i);

  std::map<LabeledAxisAccessor, Tensor> ret;
  if (qualified)
    vars = axis(i).qualified_variable_names();
  for (std::size_t i = 0; i < vars.size(); ++i)
    ret[vars[i]] = Tensor(vals[i], batch_sizes());
  return ret;
}

std::map<LabeledAxisAccessor, LabeledMatrix>
LabeledMatrix::split_subaxes(Size i, bool qualified) const
{
  auto subaxes = axis(i).subaxis_names();
  std::vector<Size> split_size;
  for (const auto & subaxis : subaxes)
    split_size.push_back(axis(i).storage_size(subaxis));

  auto vals = tensor().split(split_size, batch_dim() + i);

  std::map<LabeledAxisAccessor, LabeledMatrix> ret;
  if (qualified)
    subaxes = axis(i).qualified_subaxis_names();
  for (std::size_t i = 0; i < subaxes.size(); ++i)
    ret[subaxes[i]] = LabeledMatrix(Tensor(vals[i], batch_sizes()),
                                    {i == 0 ? &axis(0).subaxis(subaxes[i]) : &axis(0),
                                     i == 1 ? &axis(1).subaxis(subaxes[i]) : &axis(1)});
  return ret;
}

LabeledMatrix
LabeledMatrix::assemble(const TraceableTensorShape & batch_sizes,
                        const LabeledAxis & yaxis,
                        const LabeledAxis & xaxis,
                        const torch::TensorOptions & options,
                        std::vector<std::vector<Tensor>> & vals)
{
  auto rows = std::vector<Tensor>(vals.size());

  for (std::size_t i = 0; i < vals.size(); ++i)
  {
    if (!vals[i].size())
      rows[i] = Tensor::zeros(batch_sizes, {yaxis.storage_size(i), xaxis.storage_size()}, options);
    else
    {
      for (std::size_t j = 0; j < vals[i].size(); ++j)
        if (!vals[i][j].defined())
          vals[i][j] =
              Tensor::zeros(batch_sizes, {yaxis.storage_size(i), xaxis.storage_size(j)}, options);
        else
          vals[i][j] = vals[i][j].batch_expand(batch_sizes);
      rows[i] = math::base_cat(vals[i], -1);
    }
  }

  return LabeledMatrix(math::base_cat(rows, -2), {&yaxis, &xaxis});
}
} // namespace neml2
