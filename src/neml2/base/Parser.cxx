#include "neml2/base/Parser.h"

namespace neml2
{
void
Parser::parse_and_manufacture(const std::string & filename)
{
  parse(filename);
  auto & factory = Factory::get_factory();
  factory.clear();
  factory.manufacture(parameter_collection());
}

} // namespace neml2
