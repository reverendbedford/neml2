#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "SampleSecDerivModel.h"

TEST_CASE("ADSecDerivModel", "[ADSecDerivModel]")
{
  TorchSize nbatch = 10;
  auto model = SampleSecDerivModel("sample_model");
  auto ad_model = ADSampleSecDerivModel("ad_sample_model");

  LabeledVector in(nbatch, model.input());
  in.slice(0, "state").set(Scalar(1.1, nbatch), "x1");
  in.slice(0, "state").set(Scalar(0.01, nbatch), "x2");

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
