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

#include "TestUtils.h"
#include "SampleRateModel.h"
#include "neml2/models/ImplicitTimeIntegration.h"

using namespace neml2;

TEST_CASE("ImplicitTimeIntegration", "[ImplicitTimeIntegration]")
{
  TorchSize nbatch = 10;
  auto rate = std::make_shared<SampleRateModel>("sample_rate");
  auto implicit_rate = ImplicitTimeIntegration("implicit_time_integration", rate);

  SECTION("model definition")
  {
    REQUIRE(implicit_rate.input().has_subaxis("state"));
    REQUIRE(implicit_rate.input().subaxis("state").has_variable<Scalar>("foo"));
    REQUIRE(implicit_rate.input().subaxis("state").has_variable<Scalar>("bar"));
    REQUIRE(implicit_rate.input().subaxis("state").has_variable<SymR2>("baz"));

    REQUIRE(implicit_rate.input().has_subaxis("old_state"));
    REQUIRE(implicit_rate.input().subaxis("old_state").has_variable<Scalar>("foo"));
    REQUIRE(implicit_rate.input().subaxis("old_state").has_variable<Scalar>("bar"));
    REQUIRE(implicit_rate.input().subaxis("old_state").has_variable<SymR2>("baz"));

    REQUIRE(implicit_rate.input().has_subaxis("forces"));
    REQUIRE(implicit_rate.input().subaxis("forces").has_variable<Scalar>("temperature"));
    REQUIRE(implicit_rate.input().subaxis("forces").has_variable<Scalar>("time"));

    REQUIRE(implicit_rate.input().has_subaxis("old_forces"));
    REQUIRE(implicit_rate.input().subaxis("forces").has_variable<Scalar>("time"));

    REQUIRE(implicit_rate.output().has_variable("residual"));
    REQUIRE(implicit_rate.output().storage_size() == 8);
  }

  LabeledVector in(nbatch, implicit_rate.input());
  auto baz = SymR2::init(0.5, 1.1, 3.2, -1.2, 1.1, 5.9).batch_expand(nbatch);
  auto baz_old = SymR2::init(0, 0, 0, 0, 0, 0).batch_expand(nbatch);
  in.slice("state").set(Scalar(1.1, nbatch), "foo");
  in.slice("state").set(Scalar(0.01, nbatch), "bar");
  in.slice("state").set(baz, "baz");
  in.slice("old_state").set(Scalar(0, nbatch), "foo");
  in.slice("old_state").set(Scalar(0, nbatch), "bar");
  in.slice("old_state").set(baz_old, "baz");
  in.slice("forces").set(Scalar(15, nbatch), "temperature");
  in.slice("forces").set(Scalar(1.3, nbatch), "time");
  in.slice("old_forces").set(Scalar(1.1, nbatch), "time");

  SECTION("model derivatives")
  {
    auto exact = implicit_rate.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, implicit_rate.output(), implicit_rate.input());
    finite_differencing_derivative(
        [implicit_rate](const LabeledVector & x) { return implicit_rate.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor(), /*rtol=*/0, /*atol=*/5e-4));
  }

  SECTION("nonlinear system")
  {
    ImplicitModel::stage = ImplicitModel::Stage::SOLVING;
    implicit_rate.cache_input(in);
    auto x = in("state");

    auto [r, J1] = implicit_rate.residual_and_Jacobian(x);
    auto value = implicit_rate.value(in);
    REQUIRE(torch::allclose(r, value("residual")));

    BatchTensor<1> J2 = J1.clone();
    finite_differencing_derivative(
        [implicit_rate](const BatchTensor<1> & x) { return implicit_rate.residual(x); }, x, J2);
    REQUIRE(torch::allclose(J1, J2));
  }
}
