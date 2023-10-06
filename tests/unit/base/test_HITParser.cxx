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

TEST_CASE("HITParser", "[base]")
{
  SECTION("class HITParser")
  {
    HITParser parser;
    SECTION("parse")
    {
      auto all_options = parser.parse("unit/base/test_HITParser1.i");
      OptionSet options = all_options["Models"]["foo"];

      SECTION("metadata")
      {
        REQUIRE(options.get<std::string>("name") == "foo");
        REQUIRE(options.get<std::string>("type") == "SampleParserTestingModel");
      }

      SECTION("default values")
      {
        REQUIRE(options.get<std::vector<LabeledAxisAccessor>>("additional_outputs").empty());
        REQUIRE(options.get<bool>("use_AD_first_derivative") == false);
        REQUIRE(options.get<bool>("use_AD_second_derivative") == false);
      }

      SECTION("booleans")
      {
        REQUIRE(options.get<bool>("bool") == true);
        REQUIRE(options.get<std::vector<bool>>("bool_vec") ==
                std::vector<bool>{true, false, false});
        REQUIRE(options.get<std::vector<std::vector<bool>>>("bool_vec_vec") ==
                std::vector<std::vector<bool>>{
                    {true, false}, {false, true, true}, {false, false, false}});
      }

      SECTION("integers")
      {
        REQUIRE(options.get<int>("int") == 5);
        REQUIRE(options.get<std::vector<int>>("int_vec") == std::vector<int>{5, 6, 7});
        REQUIRE(options.get<std::vector<std::vector<int>>>("int_vec_vec") ==
                std::vector<std::vector<int>>{{-1, 3, -2}, {-5}});
      }

      SECTION("unsigned integers")
      {
        REQUIRE(options.get<unsigned int>("uint") == 30);
        REQUIRE(options.get<std::vector<unsigned int>>("uint_vec") ==
                std::vector<unsigned int>{1, 2, 3});
        REQUIRE(options.get<std::vector<std::vector<unsigned int>>>("uint_vec_vec") ==
                std::vector<std::vector<unsigned int>>{{555}, {123}, {1, 5, 9}});
      }

      SECTION("Reals")
      {
        REQUIRE(options.get<Real>("Real") == Approx(3.14159));
        REQUIRE_THAT(options.get<std::vector<Real>>("Real_vec"),
                     Catch::Matchers::Approx(std::vector<Real>{-111, 12, 1.1}));
        REQUIRE_THAT(options.get<std::vector<std::vector<Real>>>("Real_vec_vec")[0],
                     Catch::Matchers::Approx(std::vector<Real>{1, 3, 5}));
        REQUIRE_THAT(options.get<std::vector<std::vector<Real>>>("Real_vec_vec")[1],
                     Catch::Matchers::Approx(std::vector<Real>{2, 4, 6}));
        REQUIRE_THAT(options.get<std::vector<std::vector<Real>>>("Real_vec_vec")[2],
                     Catch::Matchers::Approx(std::vector<Real>{-3, -5, -7}));
      }

      SECTION("strings")
      {
        REQUIRE(options.get<std::string>("string") == "today");
        REQUIRE(options.get<std::vector<std::string>>("string_vec") ==
                std::vector<std::string>{"is", "a", "good", "day"});
        REQUIRE(options.get<std::vector<std::vector<std::string>>>("string_vec_vec") ==
                std::vector<std::vector<std::string>>{{"neml2", "is", "very"}, {"useful"}});
      }

      SECTION("TorchShapes")
      {
        auto shape = options.get<TorchShape>("shape");
        auto shape_vec = options.get<std::vector<TorchShape>>("shape_vec");
        auto shape_vec_vec = options.get<std::vector<std::vector<TorchShape>>>("shape_vec_vec");
        REQUIRE(shape == TorchShape{1, 2, 3, 5});
        REQUIRE(shape_vec[0] == TorchShape{1, 2, 3});
        REQUIRE(shape_vec[1] == TorchShape{2, 3});
        REQUIRE(shape_vec[2] == TorchShape{5});
        REQUIRE(shape_vec_vec[0][0] == TorchShape{2, 5});
        REQUIRE(shape_vec_vec[0][1] == TorchShape{});
        REQUIRE(shape_vec_vec[0][2] == TorchShape{3, 3});
        REQUIRE(shape_vec_vec[1][0] == TorchShape{2, 2});
        REQUIRE(shape_vec_vec[1][1] == TorchShape{1});
        REQUIRE(shape_vec_vec[1][2] == TorchShape{22});
      }
    }

    SECTION("error")
    {
      SECTION("setting a suppressed option")
      {
        REQUIRE_THROWS_WITH(
            parser.parse("unit/base/test_HITParser2.i"),
            Catch::Matchers::Contains("Option named 'suppressed_option' is suppressed, and its "
                                      "value cannot be modified."));
      }
    }
  }
}
