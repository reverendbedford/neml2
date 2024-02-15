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

#include "neml2/models/SumModel.h"

using namespace neml2;

TEST_CASE("Factory", "[base]")
{
  auto options = ScalarSumModel::expected_options();
  options.name() = "example";
  options.type() = "ScalarSumModel";
  options.set<std::vector<VariableName>>("from_var") = {VariableName("state", "A"),
                                                        VariableName("state", "substate", "B")};
  options.set<VariableName>("to_var") = VariableName("state", "outsub", "C");

  OptionCollection all_options;
  all_options["Models"]["example"] = options;

  Factory::clear();
  Factory::load(all_options);
  auto & summodel = Factory::get_object<ScalarSumModel>("Models", "example");

  std::cout << summodel.input_axis() << std::endl;
  std::cout << summodel.output_axis() << std::endl;

  REQUIRE(summodel.input_axis().has_subaxis("state"));
  REQUIRE(summodel.input_axis().subaxis("state").has_subaxis("substate"));
  REQUIRE(summodel.input_axis().subaxis("state").has_variable<Scalar>("A"));
  REQUIRE(summodel.input_axis().subaxis("state").subaxis("substate").has_variable<Scalar>("B"));

  REQUIRE(summodel.output_axis().has_subaxis("state"));
  REQUIRE(summodel.output_axis().subaxis("state").has_subaxis("outsub"));
  REQUIRE(summodel.output_axis().subaxis("state").subaxis("outsub").has_variable<Scalar>("C"));
}
