#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "SampleRateModel.h"

using namespace neml2;

TEST_CASE("ADModel", "[ADModel]")
{
  TorchSize nbatch = 10;
  auto rate = SampleRateModel("sample_rate");
  auto ad_rate = ADSampleRateModel("sample_rate");

  LabeledVector in(nbatch, rate.input());
  auto baz = SymR2::init(0.5, 1.1, 3.2, -1.2, 1.1, 5.9).batch_expand(nbatch);
  in.slice(0, "state").set(Scalar(1.1, nbatch), "foo");
  in.slice(0, "state").set(Scalar(0.01, nbatch), "bar");
  in.slice(0, "state").set(baz, "baz");
  in.slice(0, "forces").set(Scalar(15, nbatch), "temperature");

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
