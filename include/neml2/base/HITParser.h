#pragma once

#include "neml2/base/Parser.h"
#include "hit/hit.h"
#include <memory>

namespace neml2
{
/// A helper class to deserialize a file written in HIT format
class HITParser : public Parser
{
public:
  HITParser() = default;

  virtual void parse(const std::string & filename);

  /// Get the root of the parsed input file
  hit::Node * root() { return _root.get(); }

  virtual const ParameterCollection & parameter_collection() const { return _all_params; }

protected:
  /// Extract (and cast) parameters into the parameter collection
  void extract_params();

private:
  class ExtractParamsWalker : public hit::Walker
  {
  public:
    ExtractParamsWalker(ParameterSet & params)
      : _params(params)
    {
    }

    void walk(const std::string & fullpath, const std::string & nodepath, hit::Node * n) override;

  private:
    ParameterSet & _params;
  };

  /// The root node of the parsed input file
  std::unique_ptr<hit::Node> _root;

  /// Collection of parameters of the parsed input file
  ParameterCollection _all_params;
};

} // namespace neml2
