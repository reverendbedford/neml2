#include <catch2/catch.hpp>

#include "TestUtils.h"
#include "SampleRateModel.h"
#include "models/ImplicitTimeIntegration.h"
#include "models/ImplicitUpdate.h"
#include "solvers/NewtonNonlinearSolver.h"

TEST_CASE("ImplicitUpdate", "[ImplicitUpdate]")
{
  TorchSize nbatch = 10;
  auto rate = SampleRateModel("sample_rate");
  auto implicit_rate = ImplicitTimeIntegration("implicit_time_integration", rate);
  auto solver = NewtonNonlinearSolver({/*atol =*/1e-10,
                                       /*rtol =*/1e-8,
                                       /*miters =*/20,
                                       /*verbose=*/false});
  auto integrate_rate = ImplicitUpdate("time_integration", implicit_rate, solver);

  SECTION("model definition")
  {
    REQUIRE(integrate_rate.input().has_subaxis("old_state"));
    REQUIRE(integrate_rate.input().subaxis("old_state").has_variable<Scalar>("foo"));
    REQUIRE(integrate_rate.input().subaxis("old_state").has_variable<Scalar>("bar"));
    REQUIRE(integrate_rate.input().subaxis("old_state").has_variable<SymR2>("baz"));

    REQUIRE(integrate_rate.input().has_subaxis("forces"));
    REQUIRE(integrate_rate.input().subaxis("forces").has_variable<Scalar>("temperature"));
    REQUIRE(integrate_rate.input().subaxis("forces").has_variable<Scalar>("time"));

    REQUIRE(integrate_rate.input().has_subaxis("old_forces"));
    REQUIRE(integrate_rate.input().subaxis("forces").has_variable<Scalar>("time"));

    REQUIRE(integrate_rate.output().has_subaxis("state"));
    REQUIRE(integrate_rate.output().subaxis("state").has_variable<Scalar>("foo"));
    REQUIRE(integrate_rate.output().subaxis("state").has_variable<Scalar>("bar"));
    REQUIRE(integrate_rate.output().subaxis("state").has_variable<SymR2>("baz"));
  }

  LabeledVector in(nbatch, integrate_rate.input());
  auto baz_old = SymR2::init(0, 0, 0, 0, 0, 0).expand_batch(nbatch);
  in.slice(0, "old_state").set(Scalar(0, nbatch), "foo");
  in.slice(0, "old_state").set(Scalar(0, nbatch), "bar");
  in.slice(0, "old_state").set(baz_old, "baz");
  in.slice(0, "forces").set(Scalar(15, nbatch), "temperature");
  in.slice(0, "forces").set(Scalar(1.3, nbatch), "time");
  in.slice(0, "old_forces").set(Scalar(1.1, nbatch), "time");

  SECTION("model values")
  {
    auto value = integrate_rate.value(in);

    LabeledVector rate_in(nbatch, rate.input());
    rate_in.set(value("state"), "state");
    rate_in.slice(0, "forces").set(in.slice(0, "forces")("temperature"), "temperature");

    auto s_np1 = value("state");
    auto s_n = in("old_state");
    auto s_dot = rate.value(rate_in)("state");

    auto t_np1 = in.slice(0, "forces").get<Scalar>("time");
    auto t_n = in.slice(0, "old_forces").get<Scalar>("time");
    auto dt = t_np1 - t_n;
    REQUIRE(torch::allclose(s_n + s_dot * dt, value("state")));
  }

  SECTION("model derivatives")
  {
    auto exact = integrate_rate.dvalue(in);

    auto numerical = LabeledMatrix(nbatch, integrate_rate.output(), integrate_rate.input());
    finite_differencing_derivative([integrate_rate](const LabeledVector & x)
                                   { return integrate_rate.value(x); },
                                   in,
                                   numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
