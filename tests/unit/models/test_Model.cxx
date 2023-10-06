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

using namespace neml2;

TEST_CASE("Model", "[models]")
{
  SECTION("class Model")
  {
    SECTION("named_parameters")
    {
      load_model("unit/models/ImplicitUpdate.i");
      auto & model = Factory::get_object<Model>("Models", "model");
      auto params = model.named_parameters(/*recurse=*/true);

      // There are three parameters:
      auto a = "implicit_rate.rate.a";
      auto b = "implicit_rate.rate.b";
      auto c = "implicit_rate.rate.c";

      REQUIRE(params.size() == 3);
      REQUIRE(params.count(a) == 1);
      REQUIRE(params.count(b) == 1);
      REQUIRE(params.count(c) == 1);

      // Make sure modifying a parameter value _actually_ modifies its dereferenced value in the
      // model. For example, the current value of a is
      auto a_val_original = params[a].clone();
      // Now add 0.02 to it
      params[a] += 0.02;
      // Check the actual value used in the model
      auto new_params = model.named_parameters(/*recurse=*/true);
      REQUIRE(torch::allclose(a_val_original + 0.02, new_params[a]));
    }
  }
}
