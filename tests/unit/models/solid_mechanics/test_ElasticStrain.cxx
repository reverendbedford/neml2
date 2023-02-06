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
#include "neml2/models/solid_mechanics/ElasticStrain.h"

using namespace neml2;

TEST_CASE("ElasticStrain", "[ElasticStrain]")
{
  auto & factory = Factory::get_factory();
  factory.clear();

  factory.create_object("Models",
                        ElasticStrain::expected_params() +
                            ParameterSet(KS{"name", "estrain"}, KS{"type", "ElasticStrain"}));

  auto & estrain = Factory::get_object<ElasticStrain>("Models", "estrain");

  SECTION("model definition")
  {
    REQUIRE(estrain.input().has_subaxis("state"));
    REQUIRE(estrain.input().subaxis("state").has_variable<SymR2>("plastic_strain"));
    REQUIRE(estrain.input().has_subaxis("forces"));
    REQUIRE(estrain.input().subaxis("forces").has_variable<SymR2>("total_strain"));
    REQUIRE(estrain.output().has_subaxis("state"));
    REQUIRE(estrain.output().subaxis("state").has_variable<SymR2>("elastic_strain"));
  }

  SECTION("model derivatives")
  {
    TorchSize nbatch = 10;
    LabeledVector in(nbatch, estrain.input());
    auto E = SymR2::init(0.1, 0.05, 0).batch_expand(nbatch);
    auto Ep = SymR2::init(0.01, 0.01, 0).batch_expand(nbatch);
    in.slice("forces").set(E, "total_strain");
    in.slice("state").set(Ep, "plastic_strain");

    auto exact = estrain.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, estrain.output(), estrain.input());
    finite_differencing_derivative(
        [estrain](const LabeledVector & x) { return estrain.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
