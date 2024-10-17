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

using namespace neml2;

TEST_CASE("Model", "[models]")
{

  SECTION("check_input")
  {
    auto & model = reload_model("unit/models/ComposedModel3.i", "model");

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

  SECTION("variable type")
  {
    auto & model = reload_model("unit/models/ComposedModel3.i", "model");

    REQUIRE(model.variable({"forces", "t"}).type() == TensorType::kScalar);
    REQUIRE(model.variable({"forces", "temperature"}).type() == TensorType::kScalar);
    REQUIRE(model.variable({"old_forces", "t"}).type() == TensorType::kScalar);
    REQUIRE(model.variable({"old_state", "bar"}).type() == TensorType::kScalar);
    REQUIRE(model.variable({"old_state", "baz"}).type() == TensorType::kSR2);
    REQUIRE(model.variable({"old_state", "foo"}).type() == TensorType::kScalar);
    REQUIRE(model.variable({"state", "bar"}).type() == TensorType::kScalar);
    REQUIRE(model.variable({"state", "baz"}).type() == TensorType::kSR2);
    REQUIRE(model.variable({"state", "foo"}).type() == TensorType::kScalar);
    REQUIRE(model.variable({"state", "sum"}).type() == TensorType::kScalar);

    REQUIRE(utils::stringify(model.variable({"forces", "t"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.variable({"forces", "temperature"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.variable({"old_forces", "t"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.variable({"old_state", "bar"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.variable({"old_state", "baz"}).type()) == "SR2");
    REQUIRE(utils::stringify(model.variable({"old_state", "foo"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.variable({"state", "bar"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.variable({"state", "baz"}).type()) == "SR2");
    REQUIRE(utils::stringify(model.variable({"state", "foo"}).type()) == "Scalar");
    REQUIRE(utils::stringify(model.variable({"state", "sum"}).type()) == "Scalar");
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

  SECTION("profiling")
  {
    auto & model = reload_model("unit/models/chaboche.i", "implicit_rate");
    auto options = default_tensor_options().device(torch::kCUDA);
    model.to(options);

    // Trace the value method
    auto forward = [&model](torch::Tensor & x) -> std::tuple<torch::Tensor>
    { return {model.dvalue(LabeledVector(x, {&model.input_axis()}))}; };
    auto forward_jit = neml2::jit::StaticGraphFunction<std::tuple<torch::Tensor>, torch::Tensor>(
        "model.value",
        forward,
        {torch::rand({20, 50, model.input_axis().storage_size()}, options)});

    auto x = torch::rand({20, 50, model.input_axis().storage_size()}, options);
    auto [y] = forward_jit(x);
    // REQUIRE(TensorShape(y.sizes()) == TensorShape{20, 50, model.output_axis().storage_size()});
    REQUIRE(torch::allclose(y, model.dvalue(LabeledVector(x, {&model.input_axis()}))));

    forward_jit.function().graph()->dump();

    Size N = 1000;
    Size M = 100;
    x = torch::rand({N, 20, 50, model.input_axis().storage_size()}, options);

    {
      TimedSection ts("original", "model.value");
      for (Size i = 0; i < N; ++i)
        model.dvalue(LabeledVector(x.index({i}), {&model.input_axis()}));
    }
    std::cout << "original:   " << timed_sections()["model.value"]["original"] << std::endl;

    {
      torch::InferenceMode inference;
      TimedSection ts("jit warmup", "model.value");
      std::cout << "warmup ";
      for (Size i = 0; i < M; ++i)
      {
        std::cout << i << " ";
        forward_jit(x.index({i}));
      }
      std::cout << std::endl;
    }
    std::cout << "jit warmup: " << timed_sections()["model.value"]["jit warmup"] << std::endl;

    {
      torch::InferenceMode inference;
      TimedSection ts("jit", "model.value");
      for (Size i = 0; i < N; ++i)
        forward_jit(x.index({i}));
    }
    std::cout << "jit:        " << timed_sections()["model.value"]["jit"] << std::endl;

    forward_jit.function().graph()->dump();
  }
}
