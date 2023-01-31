#pragma once

#include "neml2/base/Factory.h"

namespace neml2
{
/// The base class for all parsers
class Parser
{
public:
  Parser() = default;

  /// Deserialize a file, extract parameter collection, and manufacture all objects
  virtual void parse_and_manufacture(const std::string & filename);

  /// Deserialize a file given filename
  virtual void parse(const std::string & filename) = 0;

  /// Get the parsed parameter collection
  virtual const ParameterCollection & parameter_collection() const = 0;
};

} // namespace neml2
