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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "utils.h"
#include "neml2/models/Model.h"
#include "neml2/misc/math.h"

using namespace neml2;

TEST_CASE("ParameterStore", "[models]")
{
  auto & model = reload_model("unit/models/solid_mechanics/LinearIsotropicElasticity.i", "model");
  auto batch_shape = TensorShape{5, 2};
  model.reinit(batch_shape);

  SECTION("class ParameterStore")
  {
    SECTION("named_parameters")
    {
      auto & params = model.named_parameters();

      REQUIRE(params.size() == 2);
      REQUIRE(params.has_key("E"));
      REQUIRE(params.has_key("nu"));
    }

    SECTION("get_parameter")
    {
      auto & params = model.named_parameters();
      auto & E = model.get_parameter("E");
      auto & nu = model.get_parameter("nu");

      REQUIRE(Tensor(E).batch_sizes() == TensorShape());
      REQUIRE(Tensor(nu).batch_sizes() == TensorShape());

      // Modifying the individual parameter references should affect values stored in the parameter
      // dictionary.
      E = Scalar::full({1, 2}, 1.0);
      nu = Scalar::full({5, 1}, 0.3);
      REQUIRE(Tensor(params["E"]).batch_sizes() == TensorShape{1, 2});
      REQUIRE(Tensor(params["nu"]).batch_sizes() == TensorShape{5, 1});

      // Same thing say when the user wants to use torch AD
      E.requires_grad_(true);
      nu.requires_grad_(true);
      REQUIRE(Tensor(params["E"]).requires_grad());
      REQUIRE(Tensor(params["nu"]).requires_grad());
    }
  }

  SECTION("jacrev")
  {
    // Make sure torch AD can be used to get parameter derivatives
    auto & E = model.get_parameter("E");
    auto & nu = model.get_parameter("nu");

    // First prepare some arbitrary input
    auto Ee = SR2::fill(0.09, 0.04, -0.02);
    auto x = LabeledVector(Ee, {&model.input_axis()});

    SECTION("batch mismatch")
    {
      E.requires_grad_(true);
      auto y = model.value(x);
      auto S = y.base_index(VariableName("state", "S"));
      REQUIRE_THROWS_WITH(
          math::jacrev(S, E),
          Catch::Matchers::ContainsSubstring("The batch shape of the parameter must be the "
                                             "same as the batch shape of the output"));
    }

    SECTION("Jacobians are correct")
    {
      E = Scalar::full(batch_shape, 1.0);
      nu = Scalar::full(batch_shape, 0.3);

      E.requires_grad_(true);
      nu.requires_grad_(true);
      auto y = model.value(x);
      auto S = y.base_index(VariableName("state", "S"));

      // Parameter gradients from torch AD
      const auto dS_dE_exact = math::jacrev(S, E);
      const auto dS_dnu_exact = math::jacrev(S, nu);

      // Parameter gradients from finite differencing
      const auto dS_dE_FD = finite_differencing_derivative(
          [&](const Tensor & Ep)
          {
            E = Ep;
            auto y = model.value(x);
            auto S = y.base_index(VariableName("state", "S"));
            return S;
          },
          Tensor(E).clone());
      const auto dS_dnu_FD = finite_differencing_derivative(
          [&](const Tensor & nup)
          {
            nu = nup;
            auto y = model.value(x);
            auto S = y.base_index(VariableName("state", "S"));
            return S;
          },
          Tensor(nu).clone());
      REQUIRE(torch::allclose(dS_dnu_exact, dS_dnu_FD));
    }
  }
}

TEST_CASE("Nested parameter registration")
{
  auto & model = reload_model("unit/models/test_ParameterStore.i", "model");

  const auto & params = model.named_parameters();
  REQUIRE(params.has_key("E1::value"));
  REQUIRE(params.has_key("E2::value"));
  REQUIRE(params.has_key("E3::value"));
  REQUIRE(params.has_key("elasticity1::nu"));
  REQUIRE(params.has_key("elasticity2::nu"));
  REQUIRE(params.has_key("elasticity2_another::nu"));
  REQUIRE(params.has_key("elasticity3::nu"));
}
