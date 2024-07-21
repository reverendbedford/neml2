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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "utils.h"
#include "neml2/models/Model.h"

using namespace neml2;

TEST_CASE("Model", "[models]")
{
  auto & model = reload_model("unit/models/ComposedModel3.i", "model");

  SECTION("check_input")
  {
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
    REQUIRE(model.output_type({"state", "sum"}) == TensorType::kScalar);
    REQUIRE(utils::stringify(model.output_type({"state", "sum"})) == "Scalar");
  }
}
