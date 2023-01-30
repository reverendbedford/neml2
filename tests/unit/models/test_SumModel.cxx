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
#include "neml2/models/SumModel.h"

using namespace neml2;

TEST_CASE("SumModel, Scalar", "[SumModel]")
{
  TorchSize nbatch = 10;
  auto summodel = SumModel<Scalar>(
      "example", {{"state", "A"}, {"state", "substate", "B"}}, {"state", "outsub", "C"});

  SECTION("model definition")
  {
    REQUIRE(summodel.input().has_subaxis("state"));
    REQUIRE(summodel.input().subaxis("state").has_subaxis("substate"));
    REQUIRE(summodel.input().subaxis("state").has_variable<Scalar>("A"));
    REQUIRE(summodel.input().subaxis("state").subaxis("substate").has_variable<Scalar>("B"));

    REQUIRE(summodel.output().has_subaxis("state"));
    REQUIRE(summodel.output().subaxis("state").has_subaxis("outsub"));
    REQUIRE(summodel.output().subaxis("state").subaxis("outsub").has_variable<Scalar>("C"));
  }

  LabeledVector in(nbatch, summodel.input());
  in.slice("state").slice("substate").set(Scalar(2.0, nbatch), "B");
  in.slice("state").set(Scalar(3.0, nbatch), "A");

  SECTION("model values")
  {
    auto v = summodel.value(in);
    REQUIRE(torch::allclose(v.tensor(), Scalar(5.0, nbatch)));
  }

  SECTION("model derivatives")
  {
    auto exact = summodel.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, summodel.output(), summodel.input());
    finite_differencing_derivative(
        [summodel](const LabeledVector & x) { return summodel.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}

TEST_CASE("SumModel, SymR2", "[SumModel]")
{
  TorchSize nbatch = 10;
  auto summodel = SumModel<SymR2>(
      "example", {{"state", "A"}, {"state", "substate", "B"}}, {"state", "outsub", "C"});

  SECTION("model definition")
  {
    REQUIRE(summodel.input().has_subaxis("state"));
    REQUIRE(summodel.input().subaxis("state").has_subaxis("substate"));
    REQUIRE(summodel.input().subaxis("state").has_variable<SymR2>("A"));
    REQUIRE(summodel.input().subaxis("state").subaxis("substate").has_variable<SymR2>("B"));

    REQUIRE(summodel.output().has_subaxis("state"));
    REQUIRE(summodel.output().subaxis("state").has_subaxis("outsub"));
    REQUIRE(summodel.output().subaxis("state").subaxis("outsub").has_variable<SymR2>("C"));
  }

  LabeledVector in(nbatch, summodel.input());
  in.slice("state").slice("substate").set(SymR2::init(2.0, 3.0, 4.0).batch_expand(nbatch), "B");
  in.slice("state").set(SymR2::init(1.0, 2.0, 3.0).batch_expand(nbatch), "A");

  SECTION("model values")
  {
    auto v = summodel.value(in);
    REQUIRE(torch::allclose(v.tensor(), SymR2::init(3.0, 5.0, 7.0).batch_expand(nbatch)));
  }

  SECTION("model derivatives")
  {
    auto exact = summodel.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, summodel.output(), summodel.input());
    finite_differencing_derivative(
        [summodel](const LabeledVector & x) { return summodel.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
