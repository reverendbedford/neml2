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

namespace neml2
{
namespace utils
{
std::string demangle(const char * name);

constexpr double sqrt2 = 1.4142135623730951;

inline constexpr double
mandelFactor(TorchSize i)
{
  return i < 3 ? 1.0 : sqrt2;
}

inline TorchSize
storage_size(const TorchShape & shape)
{
  TorchSize sz = 1;
  return std::accumulate(shape.begin(), shape.end(), sz, std::multiplies<TorchSize>());
}

template <typename... TorchShapeRef>
inline TorchShape
add_shapes(TorchShapeRef... shapes)
{
  TorchShape net;
  (net.insert(net.end(), shapes.begin(), shapes.end()), ...);
  return net;
}

std::string indentation(int level, int indent = 2);

template <typename T>
std::string
stringify(const T & t)
{
  std::ostringstream os;
  os << t;
  return os.str();
}

std::vector<std::string> split(const std::string & str, const std::string & delims);

std::string trim(const std::string & str, const std::string & white_space = " \t\n\v\f\r");

template <typename T>
inline T
parse(const std::string & raw_str)
{
  T val;
  std::stringstream ss(utils::trim(raw_str));
  ss >> val;
  neml_assert(!ss.fail() && ss.eof(), "parameter parsing failed");
  return val;
}

template <>
inline bool
parse<bool>(const std::string & raw_str)
{
  std::string val = parse<std::string>(raw_str);
  if (val == "true")
    return true;
  if (val == "false")
    return false;

  throw NEMLException("Failed to parse boolean value. Only 'true' and 'false' are recognized.");
}

template <typename T>
inline std::vector<T>
parse_vector(const std::string & raw_str)
{
  auto tokens = utils::split(raw_str, " \t\n\v\f\r");
  std::vector<T> ret(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++)
    ret[i] = parse<T>(tokens[i]);
  return ret;
}

template <typename T>
inline std::vector<std::vector<T>>
parse_vector_vector(const std::string & raw_str)
{
  auto token_vecs = utils::split(raw_str, ";");
  std::vector<std::vector<T>> ret(token_vecs.size());
  for (size_t i = 0; i < token_vecs.size(); i++)
    ret[i] = parse_vector<T>(token_vecs[i]);
  return ret;
}
} // namespace utils
} // namespace neml2
