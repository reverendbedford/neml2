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

#include "neml2/misc/parser_utils.h"
#include <cxxabi.h>

namespace neml2
{
const char *
ParserException::what() const throw()
{
  return _msg.c_str();
}

namespace utils
{
std::stringstream &
operator>>(std::stringstream & in, torch::Tensor & /**/)
{
  throw ParserException("Cannot parse torch::Tensor");
  return in;
}

std::string
demangle(const char * name)
{
  int status = -4;
  std::unique_ptr<char, void (*)(void *)> res{abi::__cxa_demangle(name, NULL, NULL, &status),
                                              std::free};
  return (status == 0) ? res.get() : name;
}

std::vector<std::string>
split(const std::string & str, const std::string & delims)
{
  std::vector<std::string> tokens;

  std::string::size_type last_pos = str.find_first_not_of(delims, 0);
  std::string::size_type pos = str.find_first_of(delims, std::min(last_pos + 1, str.size()));

  while (last_pos != std::string::npos)
  {
    tokens.push_back(str.substr(last_pos, pos - last_pos));
    // skip delims between tokens
    last_pos = str.find_first_not_of(delims, pos);
    if (last_pos == std::string::npos)
      break;
    pos = str.find_first_of(delims, std::min(last_pos + 1, str.size()));
  }

  return tokens;
}

std::string
trim(const std::string & str, const std::string & white_space)
{
  const auto begin = str.find_first_not_of(white_space);
  if (begin == std::string::npos)
    return ""; // no content
  const auto end = str.find_last_not_of(white_space);
  return str.substr(begin, end - begin + 1);
}

bool
start_with(std::string_view str, std::string_view prefix)
{
  return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
}

bool
end_with(std::string_view str, std::string_view suffix)
{
  return str.size() >= suffix.size() &&
         0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

template <>
bool
parse<bool>(const std::string & raw_str)
{
  std::string val = parse<std::string>(raw_str);
  if (val == "true")
    return true;
  if (val == "false")
    return false;

  throw ParserException("Failed to parse boolean value. Only 'true' and 'false' are recognized.");
}

template <>
VariableName
parse<VariableName>(const std::string & raw_str)
{
  auto tokens = split(raw_str, "/ \t\n\v\f\r");
  return VariableName(tokens);
}

template <>
TorchShape
parse<TorchShape>(const std::string & raw_str)
{
  if (!start_with(raw_str, "(") || !end_with(raw_str, ")"))
    throw ParserException("Trying to parse " + raw_str +
                          " as a shape, but a shape must start with '(' and end with ')'");

  auto inner = trim(raw_str, "() \t\n\v\f\r");
  auto tokens = split(inner, ", \t\n\v\f\r");

  TorchShape shape;
  for (auto & token : tokens)
    shape.push_back(parse<TorchSize>(token));

  return shape;
}
} // namespace utils
} // namespace neml2
