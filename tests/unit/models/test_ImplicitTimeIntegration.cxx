#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "SampleRateModel.h"
#include "models/ImplicitTimeIntegration.h"

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
  in.slice(0, "state").set(Scalar(1.1, nbatch), "foo");
  in.slice(0, "state").set(Scalar(0.01, nbatch), "bar");
  in.slice(0, "state").set(baz, "baz");
  in.slice(0, "old_state").set(Scalar(0, nbatch), "foo");
  in.slice(0, "old_state").set(Scalar(0, nbatch), "bar");
  in.slice(0, "old_state").set(baz_old, "baz");
  in.slice(0, "forces").set(Scalar(15, nbatch), "temperature");
  in.slice(0, "forces").set(Scalar(1.3, nbatch), "time");
  in.slice(0, "old_forces").set(Scalar(1.1, nbatch), "time");

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
