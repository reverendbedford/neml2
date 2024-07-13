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
class SR2;
class WR2;
class SWR4;
class SSR4;
class WSR4;
namespace math
{
constexpr Real eps = std::numeric_limits<at::scalar_value_type<Real>::type>::epsilon();

constexpr Real sqrt2 = 1.4142135623730951;
constexpr Real invsqrt2 = 0.7071067811865475;

constexpr Size mandel_reverse_index[3][3] = {{0, 5, 4}, {5, 1, 3}, {4, 3, 2}};
constexpr Size mandel_index[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

constexpr Size skew_reverse_index[3][3] = {{0, 2, 1}, {2, 0, 0}, {1, 0, 0}};
constexpr Real skew_factor[3][3] = {{0.0, -1.0, 1.0}, {1.0, 0.0, -1.0}, {-1.0, 1.0, 0.0}};

inline constexpr Real
mandel_factor(Size i)
{
  return i < 3 ? 1.0 : sqrt2;
}

/**
 * @brief A helper class to hold static data of type torch::Tensor
 *
 * This class exists because torch::Tensor cannot be declared as constexpr nor as static data in the
 * global scope. The former is obvious. The latter is because at the time static variables are
 * initialized, some torch data structures have not been properly initialized yet.
 *
 */
struct ConstantTensors
{
  ConstantTensors();

  // Get the global constants
  static ConstantTensors & get();

  static const torch::Tensor & full_to_mandel_map();
  static const torch::Tensor & mandel_to_full_map();
  static const torch::Tensor & full_to_mandel_factor();
  static const torch::Tensor & mandel_to_full_factor();
  static const torch::Tensor & full_to_skew_map();
  static const torch::Tensor & skew_to_full_map();
  static const torch::Tensor & full_to_skew_factor();
  static const torch::Tensor & skew_to_full_factor();

private:
  torch::Tensor _full_to_mandel_map;
  torch::Tensor _mandel_to_full_map;
  torch::Tensor _full_to_mandel_factor;
  torch::Tensor _mandel_to_full_factor;
  torch::Tensor _full_to_skew_map;
  torch::Tensor _skew_to_full_map;
  torch::Tensor _full_to_skew_factor;
  torch::Tensor _skew_to_full_factor;
};

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
                            Size dim = 0);

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
                            Size dim = 0);

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
BatchTensor full_to_mandel(const BatchTensor & full, Size dim = 0);

/**
 * @brief Convert a BatchTensor from Mandel notation to full notation.
 *
 * See @ref full_to_mandel for a detailed explanation.
 *
 * @param mandel The input tensor in Mandel notation
 * @param dim The base dimension where the symmetric axes start
 * @return BatchTensor The resulting tensor in full notation.
 */
BatchTensor mandel_to_full(const BatchTensor & mandel, Size dim = 0);

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
BatchTensor full_to_skew(const BatchTensor & full, Size dim = 0);

/**
 * @brief Convert a BatchTensor from skew vector notation to full notation.
 *
 * See @ref full_to_skew for a detailed explanation.
 *
 * @param skew The input tensor in skew notation
 * @param dim The base dimension where the symmetric axes start
 * @return BatchTensor The resulting tensor in full notation.
 */
BatchTensor skew_to_full(const BatchTensor & skew, Size dim = 0);

/**
 * @brief Use automatic differentiation (AD) to calculate the derivatives w.r.t. to the parameter
 *
 * @warning Torch (and hence NEML2) AD wasn't designed to compute the full Jacobian from the very
 * beginning. Using this method to calculate the full Jacobian is inefficient and is subjected to
 * some restrictions on batch shapes: This method will only work when the output \p y and the
 * paramter \p p have the same batch shape.
 *
 * However, in practice, the batch shape of the output \p y and the batch shape of the parameter \p
 * p can be different. In that case, calculating the full Jacobian is not possible, and an exception
 * will be thrown.
 *
 * One possible (inefficient) workaround is to expand and copy the parameter \p p batch dimensions,
 * e.g., batch_expand_copy, _before_ calculating the output \p y.
 *
 * @param y The `BatchTensor` to to be differentiated
 * @param p The parameter to take derivatives with respect to
 * @return BatchTensor \f$\partial y/\partial p\f$
 */
BatchTensor jacrev(const BatchTensor & y, const BatchTensor & p);

BatchTensor base_diag_embed(const BatchTensor & a, Size offset = 0, Size d1 = -2, Size d2 = -1);

/// Product w_ik e_kj - e_ik w_kj with e SR2 and w WR2
SR2 skew_and_sym_to_sym(const SR2 & e, const WR2 & w);

/// Derivative of w_ik e_kj - e_ik w_kj wrt. e
SSR4 d_skew_and_sym_to_sym_d_sym(const WR2 & w);

/// Derivative of w_ik e_kj - e_ik w_kj wrt. w
SWR4 d_skew_and_sym_to_sym_d_skew(const SR2 & e);

/// Shortcut product a_ik b_kj - b_ik a_kj with both SR2
WR2 multiply_and_make_skew(const SR2 & a, const SR2 & b);

/// Derivative of a_ik b_kj - b_ik a_kj wrt a
WSR4 d_multiply_and_make_skew_d_first(const SR2 & b);

/// Derivative of a_ik b_kj - b_ik a_kj wrt b
WSR4 d_multiply_and_make_skew_d_second(const SR2 & a);

/// Concatenate a list of BatchTensors
template <
    typename Container,
    std::enable_if_t<
        std::is_convertible_v<typename std::iterator_traits<
                                  decltype(std::declval<Container>().begin())>::iterator_category,
                              std::input_iterator_tag> &&
            std::is_convertible_v<typename std::iterator_traits<
                                      decltype(std::declval<Container>().end())>::iterator_category,
                                  std::input_iterator_tag>,
        int> = 0>
inline BatchTensor
cat(Container && tensors, Size dim)
{
  std::vector<torch::Tensor> torch_tensors(tensors.begin(), tensors.end());
  return BatchTensor(torch::cat(torch_tensors, dim), tensors.begin()->batch_dim());
}

namespace linalg
{
/// Vector norm of a vector. Falls back to math::abs is \p v is a Scalar.
BatchTensor vector_norm(const BatchTensor & v);

/// Inverse of a square matrix
BatchTensor inv(const BatchTensor & m);

/// Solve the linear system A X = B
BatchTensor solve(const BatchTensor & A, const BatchTensor & B);

std::tuple<BatchTensor, BatchTensor> lu_factor(const BatchTensor & A, bool pivot = true);

BatchTensor lu_solve(const BatchTensor & LU,
                     const BatchTensor & pivots,
                     const BatchTensor & B,
                     bool left = true,
                     bool adjoint = false);
} // namespace linalg
} // namespace math
} // namespace neml2
