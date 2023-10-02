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
  // SECTION("batched") { load_model("unit/parameters/test_parameter_partials_batched.i"); }
  // SECTION("unbatched") { load_model("unit/parameters/test_parameter_partials_unbatched.i"); }

  // auto & driver = Factory::get_object<ModelUnitTest>("Drivers", "unit");
  // auto & model = driver.model();

  // // There are three parameters:
  // auto param_a = "implicit_rate.rate.a";
  // auto param_b = "implicit_rate.rate.b";
  // auto param_c = "implicit_rate.rate.c";
  // // Let's track the derivatives of a
  // model.trace_parameters({{param_a, true}, {param_b, false}, {param_c, false}});

  // // Evaluate the model value
  // auto out = model.value(driver.in());

  // // dout/da
  // auto a = model.get_parameter<Scalar>(param_a, true);
  // auto dout_da = model.dparam(out, a);

  // // Use FD to check the parameter derivatives
  // auto a0 = a.clone().detach();
  // auto dout_da0 = finite_differencing_derivative(
  //     [&](const BatchTensor & x)
  //     {
  //       model.set_parameters({{param_a, x}});
  //       return model.value(driver.in());
  //     },
  //     a0);
  // REQUIRE(torch::allclose(dout_da, dout_da0));
}
