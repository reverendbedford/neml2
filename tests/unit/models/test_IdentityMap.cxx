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
#include "neml2/models/IdentityMap.h"

using namespace neml2;

TEST_CASE("IdentityMap, Scalar", "[IdentityMap]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        ScalarIdentityMap::expected_params() +
                            ParameterSet(KS{"name", "example"},
                                         KS{"type", "ScalarIdentityMap"},
                                         KVS{"from_var", {"force", "A"}},
                                         KVS{"to_var", {"state", "internal_state", "C"}}));

  auto & map = Factory::get_object<ScalarIdentityMap>("Models", "example");

  SECTION("model definition")
  {
    REQUIRE(map.input().has_subaxis("force"));
    REQUIRE(map.input().subaxis("force").has_variable<Scalar>("A"));

    REQUIRE(map.output().has_subaxis("state"));
    REQUIRE(map.output().subaxis("state").has_subaxis("internal_state"));
    REQUIRE(map.output().subaxis("state").subaxis("internal_state").has_variable<Scalar>("C"));
  }

  TorchSize nbatch = 10;
  LabeledVector in(nbatch, map.input());
  in.slice("force").set(Scalar(2.0, nbatch), "A");

  SECTION("model values")
  {
    auto v = map.value(in);
    REQUIRE(torch::allclose(v.tensor(), Scalar(2.0, nbatch)));
  }

  SECTION("model derivatives")
  {
    auto exact = map.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, map.output(), map.input());
    finite_differencing_derivative(
        [map](const LabeledVector & x) { return map.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}

TEST_CASE("IdentityMap, SymR2", "[IdentityMap]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        SymR2IdentityMap::expected_params() +
                            ParameterSet(KS{"name", "example"},
                                         KS{"type", "SymR2IdentityMap"},
                                         KVS{"from_var", {"force", "A"}},
                                         KVS{"to_var", {"state", "internal_state", "C"}}));

  auto & map = Factory::get_object<SymR2IdentityMap>("Models", "example");

  SECTION("model definition")
  {
    REQUIRE(map.input().has_subaxis("force"));
    REQUIRE(map.input().subaxis("force").has_variable<SymR2>("A"));

    REQUIRE(map.output().has_subaxis("state"));
    REQUIRE(map.output().subaxis("state").has_subaxis("internal_state"));
    REQUIRE(map.output().subaxis("state").subaxis("internal_state").has_variable<SymR2>("C"));
  }

  TorchSize nbatch = 10;
  LabeledVector in(nbatch, map.input());
  in.slice("force").set(SymR2::init(1.0, 2.0, 3.0).batch_expand(nbatch), "A");

  SECTION("model values")
  {
    auto v = map.value(in);
    REQUIRE(torch::allclose(v.tensor(), SymR2::init(1.0, 2.0, 3.0).batch_expand(nbatch)));
  }

  SECTION("model derivatives")
  {
    auto exact = map.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, map.output(), map.input());
    finite_differencing_derivative(
        [map](const LabeledVector & x) { return map.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
