#include <catch2/catch.hpp>

#include "neml2/base/HITParser.h"

using namespace neml2;

TEST_CASE("parse", "[HITParser]")
{
  HITParser parser;
  parser.parse("unit/base/test_HITParser.i");
}
