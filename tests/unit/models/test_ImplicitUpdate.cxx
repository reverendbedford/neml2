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
#include "neml2/models/BackwardEulerTimeIntegration.h"
#include "neml2/models/ImplicitUpdate.h"
#include "neml2/models/ComposedModel.h"
#include "neml2/solvers/NewtonNonlinearSolver.h"

using namespace neml2;

TEST_CASE("ImplicitUpdate", "[ImplicitUpdate]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Solvers",
                        NewtonNonlinearSolver::expected_params() +
                            ParameterSet(KS{"name", "newton"},
                                         KS{"type", "NewtonNonlinearSolver"},
                                         KR{"abs_tol", 1e-10},
                                         KR{"rel_tol", 1e-8},
                                         KU{"max_its", 100},
                                         KB{"verbose", false}));
  factory.create_object("Models",
                        SampleRateModel::expected_params() +
                            ParameterSet(KS{"name", "rate"}, KS{"type", "SampleRateModel"}));
  factory.create_object("Models",
                        ScalarBackwardEulerTimeIntegration::expected_params() +
                            ParameterSet(KS{"name", "integrate_foo"},
                                         KS{"type", "ScalarBackwardEulerTimeIntegration"},
                                         KVS{"rate_variable", {"foo_rate"}},
                                         KVS{"variable", {"foo"}}));
  factory.create_object("Models",
                        ScalarBackwardEulerTimeIntegration::expected_params() +
                            ParameterSet(KS{"name", "integrate_bar"},
                                         KS{"type", "ScalarBackwardEulerTimeIntegration"},
                                         KVS{"rate_variable", {"bar_rate"}},
                                         KVS{"variable", {"bar"}}));
  factory.create_object("Models",
                        SymR2BackwardEulerTimeIntegration::expected_params() +
                            ParameterSet(KS{"name", "integrate_baz"},
                                         KS{"type", "SymR2BackwardEulerTimeIntegration"},
                                         KVS{"rate_variable", {"baz_rate"}},
                                         KVS{"variable", {"baz"}}));
  factory.create_object(
      "Models",
      ComposedModel::expected_params() +
          ParameterSet(KS{"name", "implicit_rate"},
                       KS{"type", "ComposedModel"},
                       KVS{"models", {"rate", "integrate_foo", "integrate_bar", "integrate_baz"}}));
  factory.create_object("Models",
                        ImplicitUpdate::expected_params() +
                            ParameterSet(KS{"name", "model"},
                                         KS{"type", "ImplicitUpdate"},
                                         KS{"implicit_model", "implicit_rate"},
                                         KS{"solver", "newton"}));

  auto & rate = Factory::get_object<SampleRateModel>("Models", "rate");
  auto & model = Factory::get_object<ImplicitUpdate>("Models", "model");

  SECTION("model definition")
  {
    REQUIRE(model.input().has_subaxis("old_state"));
    REQUIRE(model.input().subaxis("old_state").has_variable<Scalar>("foo"));
    REQUIRE(model.input().subaxis("old_state").has_variable<Scalar>("bar"));
    REQUIRE(model.input().subaxis("old_state").has_variable<SymR2>("baz"));

    REQUIRE(model.input().has_subaxis("forces"));
    REQUIRE(model.input().subaxis("forces").has_variable<Scalar>("temperature"));
    REQUIRE(model.input().subaxis("forces").has_variable<Scalar>("time"));

    REQUIRE(model.input().has_subaxis("old_forces"));
    REQUIRE(model.input().subaxis("forces").has_variable<Scalar>("time"));

    REQUIRE(model.output().has_subaxis("state"));
    REQUIRE(model.output().subaxis("state").has_variable<Scalar>("foo"));
    REQUIRE(model.output().subaxis("state").has_variable<Scalar>("bar"));
    REQUIRE(model.output().subaxis("state").has_variable<SymR2>("baz"));
  }

  TorchSize nbatch = 1;
  LabeledVector in(nbatch, model.input());
  auto baz_old = SymR2::init(0, 0, 0, 0, 0, 0).batch_expand(nbatch);
  in.slice("old_state").set(Scalar(0, nbatch), "foo");
  in.slice("old_state").set(Scalar(0, nbatch), "bar");
  in.slice("old_state").set(baz_old, "baz");
  in.slice("forces").set(Scalar(15, nbatch), "temperature");
  in.slice("forces").set(Scalar(1.3, nbatch), "time");
  in.slice("old_forces").set(Scalar(1.1, nbatch), "time");

  SECTION("model values")
  {
    auto value = model.value(in);

    LabeledVector rate_in(nbatch, rate.input());
    rate_in.set(value("state"), "state");
    rate_in.slice("forces").set(in.slice("forces")("temperature"), "temperature");

    auto s_np1 = value("state");
    auto s_n = in("old_state");
    auto s_dot = rate.value(rate_in)("state");

    auto t_np1 = in.slice("forces").get<Scalar>("time");
    auto t_n = in.slice("old_forces").get<Scalar>("time");
    auto dt = t_np1 - t_n;
    REQUIRE(torch::allclose(s_n + s_dot * dt, value("state")));
  }

  SECTION("model derivatives")
  {
    auto exact = model.dvalue(in);

    auto numerical = LabeledMatrix(nbatch, model.output(), model.input());
    finite_differencing_derivative(
        [model](const LabeledVector & x) { return model.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
