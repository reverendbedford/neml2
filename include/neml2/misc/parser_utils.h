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

#include "neml2/misc/types.h"
#include "neml2/misc/error.h"
#include "neml2/tensors/LabeledAxis.h"
#include "neml2/base/CrossRef.h"

namespace neml2
{
class ParserException : public std::exception
{
public:
  ParserException(const std::string & msg)
    : _msg(msg)
  {
  }

  virtual const char * what() const noexcept;

private:
  std::string _msg;
};

namespace utils
{
/// This is a dummy to prevent compilers whining about not know how to >> torch::Tensor
std::stringstream & operator>>(std::stringstream & in, torch::Tensor &);

std::string demangle(const char * name);

std::vector<std::string> split(const std::string & str, const std::string & delims);

std::string trim(const std::string & str, const std::string & white_space = " \t\n\v\f\r");

bool start_with(std::string_view str, std::string_view prefix);

bool end_with(std::string_view str, std::string_view suffix);

template <typename T>
T
parse(const std::string & raw_str)
{
  T val;
  std::stringstream ss(trim(raw_str));
  ss >> val;
  if (ss.fail() || !ss.eof())
    throw ParserException("Failed to parse '" + raw_str + "' as a " + demangle(typeid(T).name()));
  return val;
}

template <typename T>
std::vector<T>
parse_vector(const std::string & raw_str)
{
  auto tokens = split(raw_str, " \t\n\v\f\r");
  std::vector<T> ret(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++)
    ret[i] = parse<T>(tokens[i]);
  return ret;
}

template <typename T>
std::vector<std::vector<T>>
parse_vector_vector(const std::string & raw_str)
{
  auto token_vecs = split(raw_str, ";");
  std::vector<std::vector<T>> ret(token_vecs.size());
  for (size_t i = 0; i < token_vecs.size(); i++)
    ret[i] = parse_vector<T>(token_vecs[i]);
  return ret;
}

// @{ template specializations for parse
template <>
bool parse<bool>(const std::string & raw_str);
template <>
TorchShape parse<TorchShape>(const std::string & raw_str);
template <>
LabeledAxisAccessor parse<LabeledAxisAccessor>(const std::string & raw_str);
// @}
} // namespace utils
} // namespace neml2
