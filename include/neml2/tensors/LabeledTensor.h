// Copyright 2023, UChicago Argonne, LLC
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

#pragma once

#include "neml2/tensors/StorageTensor.h"

namespace neml2
{
/**
 * @brief The primary data structure in NEML2 for working with labeled tensor views.
 *
 * Each LabeledTensor consists of one BatchTensor and one or more LabeledAxis. The
 * `LabeledTensor<D>` is templated on the base dimension \f$D\f$. LabeledTensor handles the
 * creation, modification, and accessing of labeled tensors.
 *
 * @tparam D The number of base dimensions
 */
template <class Derived, TorchSize D>
class LabeledTensor : public StorageTensor<D>
{
public:
  /// Default constructor
  LabeledTensor() = default;

  /// Construct from a Tensor with batch dim and array of `LabeledAxis`
  LabeledTensor(const torch::Tensor & tensor,
                TorchSize batch_dim,
                const std::array<const LabeledAxis *, D> & axes);

  /// Construct from a BatchTensor with array of `LabeledAxis`
  LabeledTensor(const BatchTensor & tensor, const std::array<const LabeledAxis *, D> & axes);

  /// Copy constructor
  LabeledTensor(const Derived & other);

  /// Setup new empty storage
  [[nodiscard]] static Derived
  empty(TorchShapeRef batch_shape,
        const std::array<const LabeledAxis *, D> & axes,
        const torch::TensorOptions & options = default_tensor_options());

  /// Setup new storage with zeros
  [[nodiscard]] static Derived
  zeros(TorchShapeRef batch_shape,
        const std::array<const LabeledAxis *, D> & axes,
        const torch::TensorOptions & options = default_tensor_options());

  /// Assignment operator
  virtual void operator=(const StorageTensor<D> & other) override;

  virtual void copy_(const BatchTensor & other) override;

  virtual void zero_() override;

  virtual BatchTensor get(const std::array<VariableName, D> & names) const override;

  virtual void set_(const std::array<VariableName, D> &, const BatchTensor &) override;

  virtual BatchTensor assemble() const override;

  virtual BatchTensor tensor() const override;

protected:
  /// Calculate slicing indices given the names on each axis
  TorchSlice slice_indices(const std::array<VariableName, D> & names) const;

  /// Calculate storage shape given the names on each axis
  TorchSlice storage_sizes(const std::array<VariableName, D> & names) const;

  /// The tensor
  BatchTensor _tensor;
};
} // namespace neml2
