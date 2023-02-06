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

using namespace neml2;

TEST_CASE("ADModel", "[ADModel]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        SampleRateModel::expected_params() +
                            ParameterSet(KS{"name", "rate"}, KS{"type", "SampleRateModel"}));
  factory.create_object("Models",
                        ADSampleRateModel::expected_params() +
                            ParameterSet(KS{"name", "ad_rate"}, KS{"type", "ADSampleRateModel"}));

  auto & rate = Factory::get_object<SampleRateModel>("Models", "rate");
  auto & ad_rate = Factory::get_object<ADSampleRateModel>("Models", "ad_rate");

  TorchSize nbatch = 10;
  LabeledVector in(nbatch, rate.input());
  auto baz = SymR2::init(0.5, 1.1, 3.2, -1.2, 1.1, 5.9).batch_expand(nbatch);
  in.slice("state").set(Scalar(1.1, nbatch), "foo");
  in.slice("state").set(Scalar(0.01, nbatch), "bar");
  in.slice("state").set(baz, "baz");
  in.slice("forces").set(Scalar(15, nbatch), "temperature");

  SECTION("model derivatives")
  {
    auto exact = rate.dvalue(in);
    auto AD = ad_rate.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, rate.output(), rate.input());
    finite_differencing_derivative(
        [rate](const LabeledVector & x) { return rate.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor(), /*rtol=*/0, /*atol=*/5e-4));
    REQUIRE(torch::allclose(AD.tensor(), numerical.tensor(), /*rtol=*/0, /*atol=*/5e-4));
  }
}
