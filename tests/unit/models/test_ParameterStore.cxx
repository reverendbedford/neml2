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

#include <catch2/catch.hpp>

#include "utils.h"
#include "neml2/models/Model.h"
#include "neml2/misc/math.h"

using namespace neml2;

TEST_CASE("ParameterStore", "[models]")
{
  load_model("unit/models/solid_mechanics/LinearIsotropicElasticity.i");
  auto & model = Factory::get_object<Model>("Models", "model");
  auto batch_shape = TorchShape{5, 2};
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
      auto & E = model.get_parameter<Scalar>("E");
      auto & nu = model.get_parameter<Scalar>("nu");

      REQUIRE(E.batch_sizes() == TorchShape());
      REQUIRE(nu.batch_sizes() == TorchShape());

      // Modifying the individual parameter references should affect values stored in the parameter
      // dictionary.
      E = E.batch_expand_copy({1, 2});
      nu = nu.batch_expand_copy({5, 1});
      REQUIRE(BatchTensor(params["E"]).batch_sizes() == TorchShape{1, 2});
      REQUIRE(BatchTensor(params["nu"]).batch_sizes() == TorchShape{5, 1});

      // Same thing say when the user wants to use torch AD
      E.requires_grad_(true);
      nu.requires_grad_(true);
      REQUIRE(BatchTensor(params["E"]).requires_grad());
      REQUIRE(BatchTensor(params["nu"]).requires_grad());
    }
  }

  SECTION("jacrev")
  {
    // Make sure torch AD can be used to get parameter derivatives
    auto & E = model.get_parameter<Scalar>("E");
    auto & nu = model.get_parameter<Scalar>("nu");

    // First prepare some arbitrary input
    auto & Ee = model.get_input_variable<SR2>(VariableName("state", "internal", "Ee"));
    Ee = SR2::fill(0.09, 0.04, -0.02);

    // The outputs of the model
    const auto & S = model.get_output_variable<SR2>(VariableName("state", "S"));

    SECTION("batch mismatch")
    {
      E.requires_grad_(true);
      model.value();
      REQUIRE_THROWS_WITH(math::jacrev(S.value(), E),
                          Catch::Matchers::Contains("The batch shape of the parameter must be the "
                                                    "same as the batch shape of the output"));
    }

    SECTION("Jacobians are correct")
    {
      E = E.batch_expand_copy(batch_shape);
      nu = nu.batch_expand_copy(batch_shape);

      E.requires_grad_(true);
      nu.requires_grad_(true);
      model.value();

      // Parameter gradients from torch AD
      const auto dS_dE_exact = math::jacrev(S.value(), E);
      const auto dS_dnu_exact = math::jacrev(S.value(), nu);

      // Parameter gradients from finite differencing
      const auto dS_dE_FD = finite_differencing_derivative(
          [&](const BatchTensor & x)
          {
            E = x;
            model.value();
            return S.value();
          },
          E.clone());
      const auto dS_dnu_FD = finite_differencing_derivative(
          [&](const BatchTensor & x)
          {
            nu = x;
            model.value();
            return S.value();
          },
          nu.clone());
      REQUIRE(torch::allclose(dS_dnu_exact, dS_dnu_FD));
    }
  }
}
