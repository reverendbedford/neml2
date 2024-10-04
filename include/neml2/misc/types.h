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

#pragma once

#include <torch/types.h>
#include <variant>

namespace neml2
{
using Real = double;
using Integer = int;
using Size = int64_t;
using TensorShape = torch::SmallVector<Size>;
using TensorShapeRef = torch::IntArrayRef;

/**
 * @brief Traceable tensor shape
 *
 * A tensor shape can be either a concrete shape or a traceable tensor. This is useful when we need
 * to trace a function graph and let it generalize to other batch shapes.
 */
struct TraceableTensorShape : public std::variant<TensorShape, torch::Tensor>
{
  using std::variant<TensorShape, torch::Tensor>::variant;

  TraceableTensorShape(TensorShapeRef shape);
  TraceableTensorShape(Size shape);
  TraceableTensorShape(const std::initializer_list<Size> & shape);

  /// @return a pointer to the traceable tensor if it is traceable, otherwise nullptr
  const torch::Tensor * traceable() const noexcept;

  /// @return the dimension, i.e., the length of this shape
  Size size() const;

  /// Slice the shape, semantically the same as ArrayRef::slice, but traceable.
  TraceableTensorShape slice(Size start, Size end) const;

  /// Chop-off the first N elements of the shape, semantically the same as ArrayRef::slice, but traceable.
  TraceableTensorShape slice(Size N) const;

  /// @return the concrete shape (without any traceable information)
  TensorShape concrete() const;

private:
  /// Make sure the traceable tensor is 1D and of type int64
  void ensure_shape() const;
};

/**
 * @brief Traceable size
 *
 * Similar to neml2::TraceableTensorShape, but only for a single dimension.
 * @see neml2::TraceableTensorShape
 */
struct TraceableSize : public std::variant<Size, torch::Tensor>
{
  using std::variant<Size, torch::Tensor>::variant;

  /// @return a pointer to the traceable size if it is traceable, otherwise nullptr
  const torch::Tensor * traceable() const noexcept;

  /// @return the concrete size (without any traceable information)
  Size concrete() const;
};

// Bring in torch::indexing
namespace indexing
{
using namespace torch::indexing;
using TensorIndices = torch::SmallVector<TensorIndex>;
using TensorIndicesRef = torch::ArrayRef<TensorIndex>;
}

/**
 * @name RAII style default tensor options
 *
 * The factory methods like `torch::arange`, `torch::ones`, `torch::zeros`, `torch::rand` etc.
 * accept a common argument to configure the properties of the tensor being created. We predefine
 * a default tensor configuration in NEML2. This default configuration is consistently used
 * throughout NEML2.
 *
 * See https://pytorch.org/cppdocs/notes/tensor_creation.html#configuring-properties-of-the-tensor
 * for more details.
 */
///@{
/// Default floating point tensor options
torch::TensorOptions & default_tensor_options();
/// Default integral tensor options
torch::TensorOptions & default_integer_tensor_options();
/// Default floating point type
torch::Dtype & default_dtype();
/// Default integral type
torch::Dtype & default_integer_dtype();
/// Default device
torch::Device & default_device();
///@}

/// @name Default tolerances
///@{
/// Machine precision
Real & machine_precision();
/// The tolerance used in various algorithms
Real & tolerance();
/// A tighter tolerance used in various algorithms
Real & tighter_tolerance();
///@}

/// Default nested buffer name separator
std::string & buffer_name_separator();
/// Default nested parameter name separator
std::string & parameter_name_separator();

/**
 * A model can be _implicit. An implicit model need to be "solved": the state variables should be
 * iteratively updated until the residual becomes zero. During the solve, we only need derivatives
 * with respect to the input state. Therefore, the model can/should avoid unnecessary computations
 * by examining whether the current evaluation is part of the solve.
 */
bool & currently_solving_nonlinear_system();
} // namespace neml2
