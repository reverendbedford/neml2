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
#include "ModelUnitTest.h"

using namespace neml2;

TEST_CASE("Use AD to get parameter partials")
{
  SECTION("Batched") { load_model("unit/parameters/test_parameter_partials_batched.i"); }
  SECTION("Unbatched") { load_model("unit/parameters/test_parameter_partials_unbatched.i"); }

  auto & driver = Factory::get_object<ModelUnitTest>("Drivers", "unit");
  auto & model = driver.model();

  // There are three parameters:
  auto param_a = "implicit_rate.rate.a";
  auto param_b = "implicit_rate.rate.b";
  auto param_c = "implicit_rate.rate.c";
  // Let's track the derivatives of a
  model.trace_parameters({{param_a, true}, {param_b, false}, {param_c, false}});

  // Evaluate the model value
  auto out = model.value(driver.in());

  // dout/da
  auto dout_da = model.dparam(out, param_a);

  // Use FD to check the parameter derivatives
  BatchTensor<1> dout_da_FD = torch::empty_like(dout_da);
  finite_differencing_derivative(
      [&](const BatchTensor<1> & x)
      {
        model.set_parameters({{param_a, x}});
        return model.value(driver.in()).tensor();
      },
      model.named_parameters(true)[param_a].detach().clone(),
      dout_da_FD);
  REQUIRE(torch::allclose(dout_da, dout_da_FD));
}
