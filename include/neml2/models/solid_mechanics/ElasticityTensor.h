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

#include "neml2/models/NonlinearParameter.h"

namespace neml2
{
/**
 * @brief Superclass for defining elasticity tensors in terms of other parameters
 */
class ElasticityTensor : public NonlinearParameter<SSR4>
{
public:
  /// Various options for parameter inputs, will need to add more as we get more tensors
  enum class ParamType
  {
    YOUNGS,
    POISSONS,
    SHEAR,
    INVALID
  };

  static OptionSet expected_options();

  ElasticityTensor(const OptionSet & options);

protected:
  /// Return the coefficients in a requested order, along with the indices to remap back to the original order
  std::tuple<std::vector<Scalar>, std::vector<size_t>>
  remap(const std::vector<ParamType> & order) const;

  /// Trivial helper function to reorder derivatives
  std::vector<Scalar> reorder_derivs(const std::vector<Scalar> & derivs,
                                     const std::vector<size_t> & order) const;

  template <std::size_t... I>
  auto vector_to_tuple_impl(const std::vector<Scalar> & v, std::index_sequence<I...>) const
  {
    return std::make_tuple(v[I]...);
  }

  template <std::size_t N>
  auto vector_to_tuple(const std::vector<Scalar> & v) const
  {
    if (v.size() != N)
    {
      throw std::runtime_error("Vector size mismatch");
    }
    return vector_to_tuple_impl(v, std::make_index_sequence<N>{});
  }

  /// Combined helper to return coefficients and derivatives in a requested order
  template <std::size_t N>
  auto combine_and_reorder(Scalar c,
                           const std::vector<Scalar> & derivs,
                           const std::vector<size_t> & order) const
  {
    auto derivs_ordered = reorder_derivs(derivs, order);
    return std::tuple_cat(std::make_tuple(c), vector_to_tuple<N>(derivs_ordered));
  }

  /// Input coefficients
  const std::vector<const Scalar *> _coef;

  /// Input coefficient types
  const std::vector<ParamType> _coef_types;
};

} // namespace neml2
