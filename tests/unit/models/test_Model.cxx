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
#include "neml2/jit/StaticGraphFunction.h"
#include "neml2/base/guards.h"

#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

using namespace neml2;

TEST_CASE("Model", "[models]")
{
  SECTION("check_input")
  {
    auto & model = reload_model("unit/models/ComposedModel3.i", "model");
    model.reinit({5, 2});

    // Input has incorrect batch shape
    auto in1 = LabeledVector::empty({2, 5}, {&model.input_axis()});
    REQUIRE_THROWS_WITH(model.value(in1),
                        Catch::Matchers::ContainsSubstring("The provided input has batch shape"));

    // Input has correct batch shape but incorrect base shape
    auto in2 = LabeledVector::empty({2}, {&model.output_axis()});
    REQUIRE_THROWS_WITH(
        model.value(in2),
        Catch::Matchers::ContainsSubstring("The provided input has base storage size"));
  }

  SECTION("input_type")
  {
    auto & model = reload_model("unit/models/ComposedModel3.i", "model");

    REQUIRE(model.input_type({"forces", "t"}) == TensorType::kScalar);
    REQUIRE(model.input_type({"forces", "temperature"}) == TensorType::kScalar);
    REQUIRE(model.input_type({"old_forces", "t"}) == TensorType::kScalar);
    REQUIRE(model.input_type({"old_state", "bar"}) == TensorType::kScalar);
    REQUIRE(model.input_type({"old_state", "baz"}) == TensorType::kSR2);
    REQUIRE(model.input_type({"old_state", "foo"}) == TensorType::kScalar);
    REQUIRE(model.input_type({"state", "bar"}) == TensorType::kScalar);
    REQUIRE(model.input_type({"state", "baz"}) == TensorType::kSR2);
    REQUIRE(model.input_type({"state", "foo"}) == TensorType::kScalar);

    REQUIRE(utils::stringify(model.input_type({"forces", "t"})) == "Scalar");
    REQUIRE(utils::stringify(model.input_type({"forces", "temperature"})) == "Scalar");
    REQUIRE(utils::stringify(model.input_type({"old_forces", "t"})) == "Scalar");
    REQUIRE(utils::stringify(model.input_type({"old_state", "bar"})) == "Scalar");
    REQUIRE(utils::stringify(model.input_type({"old_state", "baz"})) == "SR2");
    REQUIRE(utils::stringify(model.input_type({"old_state", "foo"})) == "Scalar");
    REQUIRE(utils::stringify(model.input_type({"state", "bar"})) == "Scalar");
    REQUIRE(utils::stringify(model.input_type({"state", "baz"})) == "SR2");
    REQUIRE(utils::stringify(model.input_type({"state", "foo"})) == "Scalar");
  }

  SECTION("output_type")
  {
    auto & model = reload_model("unit/models/ComposedModel3.i", "model");
    REQUIRE(model.output_type({"state", "sum"}) == TensorType::kScalar);
    REQUIRE(utils::stringify(model.output_type({"state", "sum"})) == "Scalar");
  }

  SECTION("diagnose")
  {
    SECTION("input variables")
    {
      auto & model = reload_model("unit/models/test_Model_diagnose1.i", "model");
      REQUIRE_THROWS_WITH(
          diagnose(model),
          Catch::Matchers::ContainsSubstring(
              "Input variable whatever/foo_rate must be on one of the following sub-axes"));
    }
    SECTION("output variables")
    {
      auto & model = reload_model("unit/models/test_Model_diagnose2.i", "model");
      REQUIRE_THROWS_WITH(
          diagnose(model),
          Catch::Matchers::ContainsSubstring(
              "Output variable whatever/foo must be on one of the following sub-axes"));
    }
    SECTION("statefulness")
    {
      auto & model = reload_model("unit/models/test_Model_diagnose1.i", "model");
      REQUIRE_THROWS_WITH(
          diagnose(model),
          Catch::Matchers::ContainsSubstring("Input axis has old state variable foo, but the "
                                             "corresponding output state variable doesn't exist"));
    }
    SECTION("nonlinear system")
    {
      auto & model = reload_model("unit/models/test_Model_diagnose3.i", "model");
      REQUIRE_THROWS_WITH(
          diagnose(model),
          Catch::Matchers::ContainsSubstring(
              "This model is part of a nonlinear system. At least one of the input variables is "
              "solve-dependent, so all output variables MUST be solve-dependent"));
    }
  }

  SECTION("generalizability")
  {
    SECTION("reinit")
    {
      auto & model = reload_model("unit/models/SR2LinearCombination.i", "model");

      // Trace just the reinit method
      auto forward = [&model](torch::Tensor & batch_shape) -> std::tuple<torch::Tensor>
      {
        model.reinit(batch_shape);
        return {model.output_storage()};
      };
      auto forward_jit = neml2::jit::StaticGraphFunction<std::tuple<torch::Tensor>, torch::Tensor>(
          "dummy", forward, std::make_tuple(torch::tensor({1, 1}, torch::kInt64)));

      // Allocation should be generalizable
      auto [y] = forward_jit(torch::tensor({5, 2}, torch::kInt64));
      REQUIRE(TensorShape(y.sizes()) == TensorShape{5, 2, model.output_axis().storage_size()});
    }

    SECTION("value")
    {
      auto & model = reload_model("unit/models/SR2LinearCombination.i", "model");

      // Trace the value method
      auto forward = [&model](torch::Tensor & batch_shape,
                              torch::Tensor & x) -> std::tuple<torch::Tensor>
      {
        model.reinit(batch_shape);
        return {model.value(LabeledVector(x, {&model.input_axis()}))};
      };
      auto forward_jit =
          neml2::jit::StaticGraphFunction<std::tuple<torch::Tensor>, torch::Tensor, torch::Tensor>(
              "model.value",
              forward,
              std::make_tuple(torch::tensor({1, 1}, torch::kInt64),
                              torch::ones({1, 1, model.input_axis().storage_size()})));

      // Forward operator should be generalizable
      auto x = torch::full({5, 2, model.input_axis().storage_size()}, 2.5);
      auto [y] = forward_jit(torch::tensor({5, 2}, torch::kInt64), x);
      REQUIRE(TensorShape(y.sizes()) == TensorShape{5, 2, model.output_axis().storage_size()});

      // AND give correct results
      model.reinit({5, 2});
      REQUIRE(torch::allclose(y, model.value(LabeledVector(x, {&model.input_axis()}))));

      forward_jit.function().graph()->dump();
    }

    SECTION("dvalue")
    {
      torch::jit::setTensorExprDynamicShapeFusionEnabled(true);
      auto & model = reload_model("unit/models/chaboche.i",
                                  "implicit_rate",
                                  /*enable_ad=*/false);

      // Trace the dvalue method
      auto forward = [&model](torch::Tensor & x) -> std::tuple<torch::Tensor>
      {
        model.reinit({20, 50}, /*deriv_order=*/1);
        return {model.value(LabeledVector(x, {&model.input_axis()}))};
      };
      auto forward_jit = neml2::jit::StaticGraphFunction<std::tuple<torch::Tensor>, torch::Tensor>(
          "model.dvalue",
          forward,
          std::make_tuple(torch::rand({20, 50, model.input_axis().storage_size()})));

      // // Derivative operator should be generalizable
      // auto x = torch::full({5, 2, model.input_axis().storage_size()}, 2.5);
      // auto [y] = forward_jit(torch::tensor({5, 2}, torch::kInt64), x);
      // REQUIRE(
      //     TensorShape(y.sizes()) ==
      //     TensorShape{5, 2, model.output_axis().storage_size(),
      //     model.input_axis().storage_size()});

      // // AND give correct results
      // model.reinit({5, 2}, /*deriv_order=*/1);
      // REQUIRE(torch::allclose(y, model.dvalue(LabeledVector(x, {&model.input_axis()}))));

      // Derivative operator should be generalizable
      auto x = torch::rand({20, 50, model.input_axis().storage_size()});
      // auto [y] = forward_jit(x);
      // REQUIRE(TensorShape(y.sizes()) ==
      //         TensorShape{
      //             20, 50, model.output_axis().storage_size(),
      //             model.input_axis().storage_size()});

      // AND give correct results
      // auto y_ref = model.dvalue(LabeledVector(x, {&model.input_axis()}));
      // REQUIRE(torch::allclose(y, y_ref));

      // forward_jit.function().graph()->dump();

      {
        neml2::TimedSection ts("original", "dvalue");
        torch::InferenceMode guard;
        for (size_t i = 0; i < 1000; i++)
          auto y = model.value(LabeledVector(x, {&model.input_axis()}));
      }
      std::cout << "dvalue (original): " << neml2::timed_sections()["dvalue"]["original"]
                << " ms\n";

      {
        neml2::TimedSection ts("JIT", "dvalue");
        torch::InferenceMode guard;
        for (size_t i = 0; i < 1000; i++)
          auto [y] = forward_jit(x);
      }

      std::cout << "dvalue (JIT):      " << neml2::timed_sections()["dvalue"]["JIT"] << " ms\n";

      std::cout << "torch::jit::tensorExprFuserEnabled() = " << torch::jit::tensorExprFuserEnabled()
                << std::endl;
      std::cout << "torch::jit::tensorExprDynamicShapeFusionEnabled() = "
                << torch::jit::tensorExprDynamicShapeFusionEnabled() << std::endl;
      std::cout << "torch::jit::texprReductionsEnabled() = " << torch::jit::texprReductionsEnabled()
                << std::endl;

      // forward_jit.function().graph()->dump();
    }
  }
}
