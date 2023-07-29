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

#include "neml2/base/HITParser.h"
#include "SampleParserTestingModel.h"

using namespace neml2;

TEST_CASE("parse", "[HITParser]")
{
  HITParser parser;
  auto all_params = parser.parse("unit/base/test_HITParser.i");
  ParameterSet params = all_params["Models"]["foo"];

  // meta data
  REQUIRE(params.get<std::string>("name") == "foo");
  REQUIRE(params.get<std::string>("type") == "SampleParserTestingModel");
  REQUIRE(params.get<std::vector<LabeledAxisAccessor>>("additional_outputs").empty());
  REQUIRE(params.get<bool>("use_AD_first_derivative") == false);
  REQUIRE(params.get<bool>("use_AD_second_derivative") == false);

  // booleans
  REQUIRE(params.get<bool>("bool") == true);
  REQUIRE(params.get<std::vector<bool>>("bool_vec") == std::vector<bool>{true, false, false});
  REQUIRE(
      params.get<std::vector<std::vector<bool>>>("bool_vec_vec") ==
      std::vector<std::vector<bool>>{{true, false}, {false, true, true}, {false, false, false}});

  // integers
  REQUIRE(params.get<int>("int") == 5);
  REQUIRE(params.get<std::vector<int>>("int_vec") == std::vector<int>{5, 6, 7});
  REQUIRE(params.get<std::vector<std::vector<int>>>("int_vec_vec") ==
          std::vector<std::vector<int>>{{-1, 3, -2}, {-5}});

  // unsigned integers
  REQUIRE(params.get<unsigned int>("uint") == 30);
  REQUIRE(params.get<std::vector<unsigned int>>("uint_vec") == std::vector<unsigned int>{1, 2, 3});
  REQUIRE(params.get<std::vector<std::vector<unsigned int>>>("uint_vec_vec") ==
          std::vector<std::vector<unsigned int>>{{555}, {123}, {1, 5, 9}});

  // Reals
  REQUIRE(params.get<Real>("Real") == Approx(3.14159));
  REQUIRE_THAT(params.get<std::vector<Real>>("Real_vec"),
               Catch::Matchers::Approx(std::vector<Real>{-111, 12, 1.1}));
  REQUIRE_THAT(params.get<std::vector<std::vector<Real>>>("Real_vec_vec")[0],
               Catch::Matchers::Approx(std::vector<Real>{1, 3, 5}));
  REQUIRE_THAT(params.get<std::vector<std::vector<Real>>>("Real_vec_vec")[1],
               Catch::Matchers::Approx(std::vector<Real>{2, 4, 6}));
  REQUIRE_THAT(params.get<std::vector<std::vector<Real>>>("Real_vec_vec")[2],
               Catch::Matchers::Approx(std::vector<Real>{-3, -5, -7}));

  // strings
  REQUIRE(params.get<std::string>("string") == "today");
  REQUIRE(params.get<std::vector<std::string>>("string_vec") ==
          std::vector<std::string>{"is", "a", "good", "day"});
  REQUIRE(params.get<std::vector<std::vector<std::string>>>("string_vec_vec") ==
          std::vector<std::vector<std::string>>{{"neml2", "is", "very"}, {"useful"}});
}
