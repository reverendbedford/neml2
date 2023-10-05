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

#pragma once

#include "neml2/misc/hit/parse.h"

#include <string>
#include <vector>
#include <list>
#include <map>
#include <sstream>

namespace neml2
{
namespace hit
{
inline std::string
errormsg(std::string /*fname*/, Node * /*n*/)
{
  return "";
}

template <typename T, typename... Args>
std::string
errormsg(std::string fname, Node * n, T arg, Args... args)
{
  std::stringstream ss;
  if (n && fname.size() > 0)
    ss << fname << ":" << n->line() << "." << n->column() << ": ";
  else if (fname.size() > 0)
    ss << fname << ":0: ";
  ss << arg;
  ss << errormsg("", nullptr, args...);
  return ss.str();
}

inline std::string
errormsg(Node * /*n*/)
{
  return "";
}

template <typename T, typename... Args>
std::string
errormsg(Node * n, T arg, Args... args)
{
  std::stringstream ss;
  if (n)
    ss << n->filename() << ":" << n->line() << "." << n->column() << ": ";
  ss << arg;
  ss << errormsg(nullptr, args...);
  return ss.str();
}

class BraceNode
{
public:
  std::string str(int indent = 0);
  BraceNode & append();

  inline std::vector<BraceNode> & list() { return _list; }
  inline std::string & val() { return _val; }
  inline size_t & offset() { return _offset; }
  inline size_t & len() { return _len; }

private:
  size_t _offset;
  size_t _len;
  std::string _val;
  std::vector<BraceNode> _list;
};

size_t parseBraceNode(const std::string & input, size_t start, BraceNode & n);
class BraceExpander;

class Evaler
{
public:
  virtual ~Evaler() {}
  virtual std::string eval(Field * n, const std::list<std::string> & args, BraceExpander & exp) = 0;
};

class EnvEvaler : public Evaler
{
public:
  virtual std::string
  eval(Field * n, const std::list<std::string> & args, BraceExpander & exp) override;
};

class RawEvaler : public Evaler
{
public:
  virtual std::string
  eval(Field * n, const std::list<std::string> & args, BraceExpander & exp) override;
};

class ReplaceEvaler : public Evaler
{
public:
  virtual std::string
  eval(Field * n, const std::list<std::string> & args, BraceExpander & exp) override;
};

class BraceExpander : public Walker
{
public:
  BraceExpander() {}
  void registerEvaler(const std::string & name, Evaler & ev);
  virtual void walk(const std::string & /*fullpath*/, const std::string & /*nodepath*/, Node * n);
  std::string expand(Field * n, const std::string & input);

  std::vector<std::string> used;
  std::vector<std::string> errors;

private:
  std::string expand(Field * n, BraceNode & expr);
  std::map<std::string, Evaler *> _evalers;
  ReplaceEvaler _replace;
};

} // namespace hit
} // namespace neml2
