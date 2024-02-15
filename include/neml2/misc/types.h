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

#include <cstddef>
#include <torch/torch.h>

namespace neml2
{
typedef double Real;
typedef int Integer;
typedef int64_t TorchSize;
typedef std::vector<TorchSize> TorchShape;
typedef torch::IntArrayRef TorchShapeRef;
typedef at::indexing::TensorIndex TorchIndex;
typedef std::vector<at::indexing::TensorIndex> TorchSlice;

/**
 * TODO: make the following constants configurable
 */
/// The machine precision
static constexpr Real EPS = 1E-15;
/// The tolerance used in various algorithms
static constexpr Real TOL = 1E-6;
/// A tighter tolerance used in various algorithms
static constexpr Real TOL2 = TOL * TOL;

#define _CONCAT(x, y) x##y
#define CONCAT(x, y) _CONCAT(x, y)
#define TORCH_ENUM_PREFIX torch::k
#define TORCH_DTYPE CONCAT(TORCH_ENUM_PREFIX, DTYPE)
#define TORCH_INT_DTYPE CONCAT(TORCH_ENUM_PREFIX, INT_DTYPE)

/**
 * The factory methods like `torch::arange`, `torch::ones`, `torch::zeros`, `torch::rand` etc.
 * accept a common argument to configure the properties of the tensor being created. We predefine a
 * default tensor configuration in NEML2. This default configuration is consistently used throughout
 * NEML2. This default can be configured by CMake.
 *
 * See https://pytorch.org/cppdocs/notes/tensor_creation.html#configuring-properties-of-the-tensor
 * for more details.
 */
const torch::TensorOptions default_tensor_options();

/// We similarly want to have a default integer scalar type for some types of tensors
const torch::TensorOptions default_integer_tensor_options();
} // namespace neml2
