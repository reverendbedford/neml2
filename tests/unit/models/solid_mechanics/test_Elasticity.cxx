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
#include "neml2/models/solid_mechanics/LinearElasticity.h"

using namespace neml2;

TEST_CASE("Elasticity", "[Elasticity]")
{
  TorchSize nbatch = 10;
  Scalar E = 100;
  Scalar nu = 0.3;
  SymSymR4 C = SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {E, nu});
  auto elasticity = CauchyStressFromElasticStrain("elasticity", C);

  SECTION("model definition")
  {
    REQUIRE(elasticity.input().has_subaxis("state"));
    REQUIRE(elasticity.input().subaxis("state").has_variable<SymR2>("elastic_strain"));
    REQUIRE(elasticity.output().has_subaxis("state"));
    REQUIRE(elasticity.output().subaxis("state").has_variable<SymR2>("cauchy_stress"));
  }

  SECTION("model derivatives")
  {
    LabeledVector in(nbatch, elasticity.input());
    auto Ee = SymR2::init(0.09, 0.04, 0).batch_expand(nbatch);
    in.slice("state").set(Ee, "elastic_strain");

    auto exact = elasticity.dvalue(in);
    auto numerical = LabeledMatrix(nbatch, elasticity.output(), elasticity.input());
    finite_differencing_derivative(
        [elasticity](const LabeledVector & x) { return elasticity.value(x); }, in, numerical);

    REQUIRE(torch::allclose(exact.tensor(), numerical.tensor()));
  }
}
