#pragma once

#include "hit.h"
#include <memory>

class InputParser
{
public:
  InputParser(int argc, const char * argv[]);

  hit::Node * root() { return _root.get(); }

protected:
private:
  void readInput(const char * filename);
  void replaceCLIArgs(const std::vector<std::string> & cliargs);
  void expandInput();
  std::string hitCLIFilter(const std::vector<std::string> & cliargs) const;

  std::unique_ptr<hit::Node> _root;
  std::unique_ptr<hit::Node> _cli_root;
};
