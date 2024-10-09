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
#include "neml2/misc/math.h"

namespace neml2
{
LabeledMatrix
LabeledMatrix::identity(TensorShapeRef batch_size,
                        const LabeledAxis & axis,
                        const torch::TensorOptions & options)
{
  return LabeledMatrix(Tensor::identity(batch_size, axis.size(), options), {&axis, &axis});
}

void
LabeledMatrix::fill(const LabeledMatrix & /*other*/)
{
  // neml_assert_dbg(axis(1) == other.axis(1), "Can only accumulate matrices with conformal y
  // axes"); const auto indices = axis(0).common_indices(other.axis(0), recursive); for (const auto
  // & [idxi, idxi_other] : indices)
  //   _tensor.base_index_put_({idxi}, other.tensor().base_index({idxi_other}));
}

std::map<LabeledAxisAccessor, LabeledMatrix>
LabeledMatrix::split(Size dim) const
{
  std::map<LabeledAxisAccessor, LabeledMatrix> ret;

  const auto keys = axis(dim).subaxis_names();
  const auto vals = tensor().split(axis(dim).subaxis_sizes(), batch_dim() + dim);
  for (std::size_t i = 0; i < keys.size(); ++i)
    ret[axis(dim).qualify(keys[i])] = LabeledMatrix(
        Tensor(vals[i], batch_sizes()),
        {dim == 0 ? axis(0).subaxes()[i] : &axis(0), dim == 1 ? axis(1).subaxes()[i] : &axis(1)});

  return ret;
}

std::map<LabeledAxisAccessor, std::map<LabeledAxisAccessor, Tensor>>
LabeledMatrix::disassemble() const
{
  std::map<LabeledAxisAccessor, std::map<LabeledAxisAccessor, Tensor>> ret;

  const auto yvars = axis(0).variable_names();
  const auto xvars = axis(1).variable_names();
  const auto rows = tensor().split(axis(0).variable_sizes(), -2);
  for (std::size_t i = 0; i < yvars.size(); ++i)
  {
    const auto vals = rows[i].split(axis(1).variable_sizes(), -1);
    for (std::size_t j = 0; j < xvars.size(); ++j)
      ret[axis(0).qualify(yvars[i])][axis(1).qualify(xvars[j])] = Tensor(vals[j], batch_sizes());
  }

  return ret;
}

LabeledMatrix
LabeledMatrix::assemble(
    const std::map<LabeledAxisAccessor, std::map<LabeledAxisAccessor, Tensor>> & vals_dict,
    const LabeledAxis & yaxis,
    const LabeledAxis & xaxis)
{
  const auto yvars = yaxis.variable_names();
  const auto xvars = xaxis.variable_names();

  // Assemble columns of each row
  std::vector<Tensor> rows(yvars.size());
  for (std::size_t i = 0; i < yvars.size(); ++i)
  {
    const auto vals_row = vals_dict.find(yaxis.qualify(yvars[i]));
    if (vals_row == vals_dict.end())
      continue;

    // Look up variable values from the given dictionary.
    // If a variable is not found, tensor at that position remains undefined, and all undefiend
    // tensor will later be filled with zeros.
    std::vector<Tensor> vals(xvars.size());
    for (std::size_t j = 0; j < xvars.size(); ++j)
    {
      const auto val = vals_row->second.find(xaxis.qualify(xvars[j]));
      if (val != vals_row->second.end())
      {
        neml_assert_dbg(val->second.base_dim() == 2,
                        "During matrix assembly, found a tensor associated with variables ",
                        yvars[i],
                        "/",
                        xvars[j],
                        " with base dimension ",
                        val->second.base_dim(),
                        ". Expected base dimension of 2.");
        vals[j] = val->second;
      }
    }

    // Broadcast batch shape and find the common dtype and device.
    const auto batch_sizes = utils::broadcast_batch_sizes(vals);
    const auto options =
        torch::TensorOptions().dtype(utils::same_dtype(vals)).device(utils::same_device(vals));

    // Expand defined tensors with the broadcast batch shape and fill undefined tensors with zeros.
    for (std::size_t j = 0; j < xvars.size(); ++j)
      if (vals[j].defined())
        vals[j] = vals[j].batch_expand(batch_sizes);
      else
        vals[j] = Tensor::zeros(
            batch_sizes, {yaxis.variable_sizes()[i], xaxis.variable_sizes()[j]}, options);

    rows[i] = math::base_cat(vals, -1);
  }

  // Assemble rows
  // Broadcast batch shape and find the common dtype and device.
  const auto batch_sizes = utils::broadcast_batch_sizes(rows);
  const auto options =
      torch::TensorOptions().dtype(utils::same_dtype(rows)).device(utils::same_device(rows));

  // Expand defined tensors with the broadcast batch shape and fill undefined tensors with zeros.
  for (std::size_t i = 0; i < yvars.size(); ++i)
    if (rows[i].defined())
      rows[i] = rows[i].batch_expand(batch_sizes);
    else
      rows[i] = Tensor::zeros(batch_sizes, {yaxis.variable_sizes()[i], xaxis.size()}, options);

  return LabeledMatrix(math::base_cat(rows, -2), {&yaxis, &xaxis});
}
} // namespace neml2
