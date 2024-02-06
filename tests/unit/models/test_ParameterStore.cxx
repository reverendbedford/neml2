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
  load_model("unit/models/ImplicitUpdate.i");
  auto & model = Factory::get_object<Model>("Models", "model");
  auto batch_shape = TorchShape{5, 2};
  model.reinit(batch_shape);

  SECTION("class ParameterStore")
  {
    SECTION("named_parameters")
    {
      auto & params = model.named_parameters();

      REQUIRE(params.size() == 3);
      REQUIRE(params.has_key("rate.a"));
      REQUIRE(params.has_key("rate.b"));
      REQUIRE(params.has_key("rate.c"));
    }

    SECTION("get_parameter")
    {
      auto & params = model.named_parameters();
      auto & a = model.get_parameter<Scalar>("rate.a");
      auto & b = model.get_parameter<Scalar>("rate.b");
      auto & c = model.get_parameter<Scalar>("rate.c");

      REQUIRE(a.batch_sizes() == TorchShape());
      REQUIRE(b.batch_sizes() == TorchShape());
      REQUIRE(c.batch_sizes() == TorchShape());

      // Modifying the individual parameter references should affect values stored in the parameter
      // dictionary.
      a = a.batch_expand_copy({1, 2});
      b = b.batch_expand_copy({5, 1});
      c = c.batch_expand_copy({5, 2});
      REQUIRE(BatchTensor(params["rate.a"]).batch_sizes() == TorchShape{1, 2});
      REQUIRE(BatchTensor(params["rate.b"]).batch_sizes() == TorchShape{5, 1});
      REQUIRE(BatchTensor(params["rate.c"]).batch_sizes() == TorchShape{5, 2});

      // Same thing say when the user wants to use torch AD
      a.requires_grad_(true);
      c.requires_grad_(true);
      REQUIRE(BatchTensor(params["rate.a"]).requires_grad());
      REQUIRE(!BatchTensor(params["rate.b"]).requires_grad());
      REQUIRE(BatchTensor(params["rate.c"]).requires_grad());
    }
  }

  SECTION("jacrev")
  {
    // Make sure torch AD can be used to get parameter derivatives
    auto & a = model.get_parameter<Scalar>("rate.a");
    auto & b = model.get_parameter<Scalar>("rate.b");
    auto & c = model.get_parameter<Scalar>("rate.c");

    // First prepare some arbitrary input
    using vecstr = std::vector<std::string>;
    auto & foo_n = model.get_input_variable(vecstr{"old_state", "foo"});
    auto & bar_n = model.get_input_variable(vecstr{"old_state", "bar"});
    auto & baz_n = model.get_input_variable(vecstr{"old_state", "baz"});
    auto & T = model.get_input_variable(vecstr{"forces", "temperature"});
    auto & t = model.get_input_variable(vecstr{"forces", "t"});
    auto & t_n = model.get_input_variable(vecstr{"old_forces", "t"});

    foo_n = Scalar::full(0);
    bar_n = Scalar::full(0);
    baz_n = SR2::fill(0);
    T = Scalar::full(15);
    t = Scalar::full(1.3);
    t_n = Scalar::full(1.1);

    // The outputs of the model
    const auto & foo = model.get_output_variable(vecstr{"state", "foo"});
    // const auto & bar = model.get_output_variable(vecstr{"state", "bar"});
    // const auto & baz = model.get_output_variable(vecstr{"state", "baz"});

    SECTION("batch mismatch")
    {
      a.requires_grad_(true);
      model.value();
      REQUIRE_THROWS_WITH(math::jacrev(foo.value(), a),
                          Catch::Matchers::Contains("The batch shape of the parameter must be the "
                                                    "same as the batch shape of the output"));
    }

    SECTION("Jacobians are correct")
    {
      a = a.batch_expand_copy(batch_shape);
      b = b.batch_expand_copy(batch_shape);
      c = c.batch_expand_copy(batch_shape);
      a.requires_grad_(true);
      model.value();
      std::cout << math::jacrev(foo.value(), a) << std::endl;
    }
  }
}
