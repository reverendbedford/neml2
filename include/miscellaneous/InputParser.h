#pragma once

#include "hit.h"
#include <memory>

/// A helper class to deserialize a file written in HIT format
class InputParser
{
public:
  /// Construct an `InputParser` from a string
  InputParser(const char * fname, const std::vector<std::string> & cliargs = {});

  /// Construct an `InputParser` from commandline
  InputParser(int argc, const char * argv[]);

  hit::Node * root() { return _root.get(); }

protected:
private:
  /// Deserialize a file given filename
  void parse(const char * filename);

  /// Use commandline arguments to override the parsed file
  void replace_cliargs(const std::vector<std::string> & cliargs);

  /// An input file may contain brace expressions. This method expands those brace expressions in place.
  void expand_input();

  /// A helper to extract cliargs that will be used in `replace_cliargs`
  std::string filter_hit_cliargs(const std::vector<std::string> & cliargs) const;

  /// The root node of the parsed input. This should include all information from the input file.
  std::unique_ptr<hit::Node> _root;

  /// The root node of the parsed cliargs. This will be merged into the `_root` node.
  std::unique_ptr<hit::Node> _cli_root;
};
