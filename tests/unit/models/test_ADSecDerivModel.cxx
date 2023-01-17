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
#include "SampleSecDerivModel.h"

using namespace neml2;

TEST_CASE("ADSecDerivModel", "[ADSecDerivModel]")
{
  torch::cuda::is_available();

  TorchSize nbatch = 10;
  auto model = SampleSecDerivModel("sample_model");
  auto ad_model = ADSampleSecDerivModel("ad_sample_model");

  LabeledVector in(nbatch, model.input());
  in.slice("state").set(Scalar(1.1, nbatch), "x1");
  in.slice("state").set(Scalar(0.01, nbatch), "x2");

  SECTION("model derivatives")
  {
    auto exact = model.dvalue(in);
    auto AD = ad_model.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, model.output(), model.input());
    finite_differencing_derivative(
        [model](const LabeledVector & x) { return model.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor(), /*rtol=*/0, /*atol=*/5e-4));
    REQUIRE(torch::allclose(AD.tensor(), numerical.tensor(), /*rtol=*/0, /*atol=*/5e-4));
  }

  SECTION("model second derivatives")
  {
    auto exact = model.d2value(in);
    auto AD = ad_model.d2value(in);
    auto numerical = LabeledTensor<1, 3>(nbatch, model.output(), model.input(), model.input());
    finite_differencing_derivative(
        [model](const LabeledVector & x) { return model.dvalue(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor(), /*rtol=*/0, /*atol=*/5e-4));
    REQUIRE(torch::allclose(AD.tensor(), numerical.tensor(), /*rtol=*/0, /*atol=*/5e-4));
  }
}
