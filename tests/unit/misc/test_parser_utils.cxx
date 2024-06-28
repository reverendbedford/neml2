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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "neml2/misc/parser_utils.h"

using namespace neml2;

TEST_CASE("parser_utils", "[misc]")
{
  SECTION("split")
  {
    REQUIRE(utils::split("a/b/c", "/") == std::vector<std::string>{"a", "b", "c"});
    REQUIRE(utils::split("/b/c", "/") == std::vector<std::string>{"b", "c"});
    REQUIRE(utils::split("a/b/", "/") == std::vector<std::string>{"a", "b"});
    REQUIRE(utils::split("/", "/") == std::vector<std::string>{});
  }

  SECTION("trim")
  {
    REQUIRE(utils::trim("a b cde   ") == "a b cde");
    REQUIRE(utils::trim("(123)", "()") == "123");
  }

  SECTION("start_with")
  {
    REQUIRE(utils::start_with("a b cde   ", "a b"));
    REQUIRE(!utils::start_with("abcde", "a b"));
  }

  SECTION("end_with")
  {
    REQUIRE(utils::end_with("a b cde   ", "e   "));
    REQUIRE(!utils::end_with("abcde", "e   "));
  }

  SECTION("parse")
  {
    SECTION("torch::Tensor")
    {
      REQUIRE_THROWS_WITH(utils::parse<torch::Tensor>("1"),
                          Catch::Matchers::ContainsSubstring("Cannot parse torch::Tensor"));
    }

    SECTION("TorchShape")
    {
      REQUIRE(utils::parse<TorchShape>("(1,2,3,4,5,6)") == TorchShape{1, 2, 3, 4, 5, 6});
      REQUIRE(utils::parse<TorchShape>("(1,2,3)") == TorchShape{1, 2, 3});
      REQUIRE(utils::parse<TorchShape>("(1,2,3,)") == TorchShape{1, 2, 3});
      REQUIRE(utils::parse<TorchShape>("(,1,2,3)") == TorchShape{1, 2, 3});
      REQUIRE(utils::parse<TorchShape>("(,1,2,3,)") == TorchShape{1, 2, 3});
      REQUIRE(utils::parse<TorchShape>("( ,  1, 2, 3 , )") == TorchShape{1, 2, 3});
      REQUIRE(utils::parse<TorchShape>("()") == TorchShape{});
      REQUIRE_THROWS_WITH(
          utils::parse<TorchShape>("1"),
          Catch::Matchers::ContainsSubstring("a shape must start with '(' and end with ')'"));
    }

    SECTION("bool")
    {
      REQUIRE(utils::parse<bool>("true"));
      REQUIRE(!utils::parse<bool>("false"));
      REQUIRE_THROWS_WITH(utils::parse<bool>("off"),
                          Catch::Matchers::ContainsSubstring("Failed to parse boolean value"));
    }
  }

  SECTION("parse_csv")
  {
    std::string test_file = "unit/misc/test_parser_utils.csv";

    SECTION("filename only")
    {
      auto tensor = utils::parse_csv(test_file);
      auto correct = torch::tensor({{1., 2., 3., 4., 1., 2., 3., 1., 2., 3., 4., 5., 6.},
                                    {2., -1., -3., -5., 1., 2., 3., 1., 2., 3., 4., 5., 6.},
                                    {1.1, 1., 5., 33., 1., 2., 3., 1., 2., 3., 4., 5., 6.},
                                    {1e3, -1.2e2, 100., 33., 1., 2., 3., 1., 2., 3., 4., 5., 6.},
                                    {0., 0., 1., -2., 1., 2., 3., 1., 2., 3., 4., 5., 6.}},
                                   default_tensor_options());
      REQUIRE(torch::allclose(tensor, correct));
    }

    SECTION("filename and row indexing")
    {
      auto tensor = utils::parse_csv(test_file + "[:3]");
      auto correct = torch::tensor({{1., 2., 3., 4., 1., 2., 3., 1., 2., 3., 4., 5., 6.},
                                    {2., -1., -3., -5., 1., 2., 3., 1., 2., 3., 4., 5., 6.},
                                    {1.1, 1., 5., 33., 1., 2., 3., 1., 2., 3., 4., 5., 6.}},
                                   default_tensor_options());
      REQUIRE(torch::allclose(tensor, correct));
    }

    SECTION("filename and row indexing and column indexing")
    {
      auto tensor = utils::parse_csv(test_file + "[:3,[4,5,6]]");
      auto correct =
          torch::tensor({{1., 2., 3.}, {1., 2., 3.}, {1., 2., 3.}}, default_tensor_options());
      REQUIRE(torch::allclose(tensor, correct));
    }

    SECTION("column name substitution")
    {
      auto [filename, rows, cols] = utils::parse_csv_spec(test_file + "[1:4,sr2_0:]");
      std::cout << filename << std::endl;
      std::cout << rows << std::endl;
      std::cout << cols << std::endl;
      auto tensor = utils::parse_csv(test_file + "[1:4,sr2_0:]");
      // REQUIRE(torch::allclose(tensor, correct));
    }
  }

  SECTION("parse_csv_spec")
  {
    std::string test_file = "unit/misc/test_parser_utils.csv";

    SECTION("filename only")
    {
      auto [filename, rows, cols] = utils::parse_csv_spec(test_file);
      REQUIRE(filename == test_file);
      REQUIRE(rows.is_slice());
      REQUIRE(rows.slice().start().expect_int() == 0);
      REQUIRE(rows.slice().stop().expect_int() == torch::indexing::INDEX_MAX);
      REQUIRE(rows.slice().step().expect_int() == 1);
      REQUIRE(cols.is_slice());
      REQUIRE(cols.slice().start().expect_int() == 0);
      REQUIRE(cols.slice().stop().expect_int() == torch::indexing::INDEX_MAX);
      REQUIRE(cols.slice().step().expect_int() == 1);
    }

    SECTION("filename and row indexing")
    {
      auto [filename, rows, cols] = utils::parse_csv_spec(test_file + "[2:5:-1]");
      REQUIRE(filename == test_file);
      REQUIRE(rows.is_slice());
      REQUIRE(rows.slice().start().expect_int() == 2);
      REQUIRE(rows.slice().stop().expect_int() == 5);
      REQUIRE(rows.slice().step().expect_int() == -1);
      REQUIRE(cols.is_slice());
      REQUIRE(cols.slice().start().expect_int() == 0);
      REQUIRE(cols.slice().stop().expect_int() == torch::indexing::INDEX_MAX);
      REQUIRE(cols.slice().step().expect_int() == 1);
    }

    SECTION("filename and row indexing and column indexing")
    {
      auto [filename, rows, cols] = utils::parse_csv_spec(test_file + "[[5,6,2],5:8]");
      REQUIRE(filename == test_file);
      REQUIRE(rows.is_tensor());
      REQUIRE(
          torch::all(rows.tensor() == torch::tensor({5, 6, 2}, default_integer_tensor_options()))
              .item<bool>());
      REQUIRE(cols.is_slice());
      REQUIRE(cols.slice().start().expect_int() == 5);
      REQUIRE(cols.slice().stop().expect_int() == 8);
      REQUIRE(cols.slice().step().expect_int() == 1);
    }

    SECTION("column name substitution")
    {
      auto [filename, rows, cols] = utils::parse_csv_spec(test_file + "[::-2,foo:baz]");
      REQUIRE(filename == test_file);
      REQUIRE(rows.is_slice());
      REQUIRE(rows.slice().start().expect_int() == torch::indexing::INDEX_MAX);
      REQUIRE(rows.slice().stop().expect_int() == torch::indexing::INDEX_MIN);
      REQUIRE(rows.slice().step().expect_int() == -2);
      REQUIRE(cols.is_slice());
      REQUIRE(cols.slice().start().expect_int() == 0);
      REQUIRE(cols.slice().stop().expect_int() == 2);
      REQUIRE(cols.slice().step().expect_int() == 1);
    }

    SECTION("error")
    {
      REQUIRE_THROWS_WITH(utils::parse_csv_spec(test_file + "[]"),
                          Catch::Matchers::ContainsSubstring("Empty CSV indexing not allowed"));
      REQUIRE_THROWS_WITH(utils::parse_csv_spec(test_file + "[[123,3,2,a:b]"),
                          Catch::Matchers::ContainsSubstring("Missing closing ] in "));
      REQUIRE_THROWS_WITH(
          utils::parse_csv_spec(test_file + "[[123,3,2]:a:b]"),
          Catch::Matchers::ContainsSubstring("Expected comma after row indexing, got ':'"));
      REQUIRE_THROWS_WITH(
          utils::parse_csv_spec(test_file + "[,a:b]"),
          Catch::Matchers::ContainsSubstring("Row indexing cannot begin with comma"));
    }
  }

  SECTION("parse_indexing")
  {
    SECTION("single element indexing")
    {
      auto idx = utils::parse_indexing("35");
      REQUIRE(idx.is_integer());
      REQUIRE(idx.integer().expect_int() == 35);
    }

    SECTION("slice [start:]")
    {
      auto idx = utils::parse_indexing("32:");
      REQUIRE(idx.is_slice());
      REQUIRE(idx.slice().start().expect_int() == 32);
      REQUIRE(idx.slice().stop().expect_int() == torch::indexing::INDEX_MAX);
      REQUIRE(idx.slice().step().expect_int() == 1);
    }

    SECTION("slice [:stop]")
    {
      auto idx = utils::parse_indexing(":11");
      REQUIRE(idx.is_slice());
      REQUIRE(idx.slice().start().expect_int() == 0);
      REQUIRE(idx.slice().stop().expect_int() == 11);
      REQUIRE(idx.slice().step().expect_int() == 1);
    }

    SECTION("slice [::step]")
    {
      auto idx = utils::parse_indexing("::-2");
      REQUIRE(idx.is_slice());
      REQUIRE(idx.slice().start().expect_int() == torch::indexing::INDEX_MAX);
      REQUIRE(idx.slice().stop().expect_int() == torch::indexing::INDEX_MIN);
      REQUIRE(idx.slice().step().expect_int() == -2);
    }

    SECTION("slice [start:stop]")
    {
      auto idx = utils::parse_indexing("5:-1");
      REQUIRE(idx.is_slice());
      REQUIRE(idx.slice().start().expect_int() == 5);
      REQUIRE(idx.slice().stop().expect_int() == -1);
      REQUIRE(idx.slice().step().expect_int() == 1);
    }

    SECTION("slice [start::step]")
    {
      auto idx = utils::parse_indexing("5::-1");
      REQUIRE(idx.is_slice());
      REQUIRE(idx.slice().start().expect_int() == 5);
      REQUIRE(idx.slice().stop().expect_int() == torch::indexing::INDEX_MIN);
      REQUIRE(idx.slice().step().expect_int() == -1);
    }

    SECTION("slice [:stop:step]")
    {
      auto idx = utils::parse_indexing(":2:-5");
      REQUIRE(idx.is_slice());
      REQUIRE(idx.slice().start().expect_int() == torch::indexing::INDEX_MAX);
      REQUIRE(idx.slice().stop().expect_int() == 2);
      REQUIRE(idx.slice().step().expect_int() == -5);
    }

    SECTION("slice [start:stop:step]")
    {
      auto idx = utils::parse_indexing("-5:-1:2");
      REQUIRE(idx.is_slice());
      REQUIRE(idx.slice().start().expect_int() == -5);
      REQUIRE(idx.slice().stop().expect_int() == -1);
      REQUIRE(idx.slice().step().expect_int() == 2);
    }

    SECTION("advanced indexing")
    {
      auto idx = utils::parse_indexing("[1,5,2]");
      REQUIRE(idx.is_tensor());
      REQUIRE(torch::all(idx.tensor() == torch::tensor({1, 5, 2}, default_integer_tensor_options()))
                  .item<bool>());
    }

    SECTION("error")
    {
      REQUIRE_THROWS_WITH(utils::parse_indexing("[2,3,5"),
                          Catch::Matchers::ContainsSubstring("Missing closing ] in "));
    }
  }
}
