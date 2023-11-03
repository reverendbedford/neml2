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

#include "neml2/tensors/BatchTensor.h"

namespace neml2
{
namespace math
{
constexpr Real eps = std::numeric_limits<at::scalar_value_type<Real>::type>::epsilon();

constexpr Real sqrt2 = 1.4142135623730951;
constexpr Real invsqrt2 = 0.7071067811865475;

constexpr TorchSize mandel_reverse_index[3][3] = {{0, 5, 4}, {5, 1, 3}, {4, 3, 2}};
constexpr TorchSize mandel_index[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

constexpr TorchSize skew_reverse_index[3][3] = {{0, 2, 1}, {2, 0, 0}, {1, 0, 0}};
constexpr Real skew_factor[3][3] = {{0.0, -1.0, 1.0}, {1.0, 0.0, -1.0}, {-1.0, 1.0, 0.0}};

const torch::Tensor full_to_mandel_map = torch::tensor({0, 4, 8, 5, 2, 1});
const torch::Tensor mandel_to_full_map = torch::tensor({0, 5, 4, 5, 1, 3, 4, 3, 2});
const torch::Tensor full_to_mandel_factor = torch::tensor({1.0, 1.0, 1.0, sqrt2, sqrt2, sqrt2});
const torch::Tensor mandel_to_full_factor =
    torch::tensor({1.0, invsqrt2, invsqrt2, invsqrt2, 1.0, invsqrt2, invsqrt2, invsqrt2, 1.0});

inline constexpr Real
mandel_factor(TorchSize i)
{
  return i < 3 ? 1.0 : sqrt2;
}

const torch::Tensor full_to_skew_map = torch::tensor({7, 2, 3});
const torch::Tensor skew_to_full_map = torch::tensor({0, 2, 1, 2, 0, 0, 1, 0, 0});
const torch::Tensor full_to_skew_factor = torch::tensor({1.0, 1.0, 1.0});
const torch::Tensor skew_to_full_factor =
    torch::tensor({0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0});

/**
 * @brief Generic function to reduce two axes to one with some map
 *
 * The tensor in full notation \p full can have arbitrary batch shape. The optional argument \p dim
 * denotes the base dimension starting from which the conversion should take place.
 *
 * The function will reduce the two axis at the desired location down to one, using the provided
 * maps.
 *
 * For example, a full tensor has shape `(2, 3, 1, 5; 2, 9, 3, 3, 2, 3)` where the semicolon
 * separates batch and base shapes. The *reduction* axes have base dim 2 and 3. After applying the
 * reduction, the resulting tensor will have shape `(2, 3, 1, 5; 2, 9, X, 2, 3)` where X
 * is the reduced shape. In this example, the base dim (the
 * second argument to this function) should be 2.
 *
 * @param full The input tensor in full notation
 * @param rmap The reduction map
 * @param rfactors The reduction factors
 * @param dim The base dimension where the reduced axes start
 * @return BatchTensor The reduced tensor
 */
BatchTensor full_to_reduced(const BatchTensor & full,
                            const torch::Tensor & rmap,
                            const torch::Tensor & rfactors,
                            TorchSize dim = 0);

/**
 * @brief Convert a BatchTensor from reduced notation to full notation.
 *
 * See @ref full_to_reduced for a detailed explanation.
 *
 * @param reduced The input tensor in reduced notation
 * @param rmap The unreduction map
 * @param rfactors The unreduction factors
 * @param dim The base dimension where the reduced axes start
 * @return BatchTensor The resulting tensor in full notation.
 */
BatchTensor reduced_to_full(const BatchTensor & reduced,
                            const torch::Tensor & rmap,
                            const torch::Tensor & rfactors,
                            TorchSize dim = 0);

/**
 * @brief Convert a `BatchTensor` from full notation to Mandel notation.
 *
 * The tensor in full notation \p full can have arbitrary batch shape. The optional argument \p dim
 * denotes the base dimension starting from which the conversion should take place.
 *
 * For example, a full tensor has shape `(2, 3, 1, 5; 2, 9, 3, 3, 2, 3)` where the semicolon
 * separates batch and base shapes. The *symmetric* axes have base dim 2 and 3. After converting to
 * Mandel notation, the resulting tensor will have shape `(2, 3, 1, 5; 2, 9, 6, 2, 3)`. Note how the
 * shape of the symmetric dimensions `(3, 3)` becomes `(6)`. In this example, the base dim (the
 * second argument to this function) should be 2.
 *
 * @param full The input tensor in full notation
 * @param dim The base dimension where the symmetric axes start
 * @return BatchTensor The resulting tensor using Mandel notation to represent the symmetric axes.
 */
BatchTensor full_to_mandel(const BatchTensor & full, TorchSize dim = 0);

/**
 * @brief Convert a BatchTensor from Mandel notation to full notation.
 *
 * See @ref full_to_mandel for a detailed explanation.
 *
 * @param mandel The input tensor in Mandel notation
 * @param dim The base dimension where the symmetric axes start
 * @return BatchTensor The resulting tensor in full notation.
 */
BatchTensor mandel_to_full(const BatchTensor & mandel, TorchSize dim = 0);

/**
 * @brief Convert a `BatchTensor` from full notation to skew vector notation.
 *
 * The tensor in full notation \p full can have arbitrary batch shape. The optional argument \p dim
 * denotes the base dimension starting from which the conversion should take place.
 *
 * For example, a full tensor has shape `(2, 3, 1, 5; 2, 9, 3, 3, 2, 3)` where the semicolon
 * separates batch and base shapes. The *symmetric* axes have base dim 2 and 3. After converting to
 * skew notation, the resulting tensor will have shape `(2, 3, 1, 5; 2, 9, 3, 2, 3)`. Note how the
 * shape of the symmetric dimensions `(3, 3)` becomes `(3)`. In this example, the base dim (the
 * second argument to this function) should be 2.
 *
 * @param full The input tensor in full notation
 * @param dim The base dimension where the symmetric axes start
 * @return BatchTensor The resulting tensor using skew notation to represent the skew-symmetric
 * axes.
 */
BatchTensor full_to_skew(const BatchTensor & full, TorchSize dim = 0);

/**
 * @brief Convert a BatchTensor from skew vector notation to full notation.
 *
 * See @ref full_to_skew for a detailed explanation.
 *
 * @param skew The input tensor in skew notation
 * @param dim The base dimension where the symmetric axes start
 * @return BatchTensor The resulting tensor in full notation.
 */
BatchTensor skew_to_full(const BatchTensor & skew, TorchSize dim = 0);

/**
 * @brief Use automatic differentiation to calculate the derivatives w.r.t. to the parameter
 * _component by component_
 *
 * If the parameter is batched, the batch shape must be the _same_ with the batch shape of the
 * output \p o. Note that this is the poor-man's attempt at obtaining the Jacobian -- libTorch
 * really isn't designed for this -- unless we have vmap...
 *
 * @param o The `BatchTensor` to to be differentiated
 * @param p The parameter to take derivatives with respect to
 * @return BatchTensor $\partial o/\partial p$
 */
BatchTensor jacrev(const BatchTensor & o, const BatchTensor & p);

} // namespace math
} // namespace neml2
