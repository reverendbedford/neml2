#include <catch2/catch.hpp>

#include "misc/InputParser.h"
#include "misc/Registry.h"

#include <iostream>
#include <fstream>
#include <streambuf>

using namespace neml2;

TEST_CASE("HIT parser", "[HIT]")
{
  std::string fname = "unit/misc/test_hit.i";
  InputParser parser(fname.c_str());

  SECTION("hellow world")
  {
    std::ifstream gold(fname);
    REQUIRE(gold.is_open());
    std::string correct((std::istreambuf_iterator<char>(gold)), std::istreambuf_iterator<char>());

    std::ostringstream oss;
    oss << parser.root().render() << std::endl;
    std::string mine = oss.str();

    REQUIRE(mine == correct);
  }
}
