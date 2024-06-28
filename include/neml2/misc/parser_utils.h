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
#include "neml2/tensors/Variable.h"
#include "neml2/base/CrossRef.h"
#include "neml2/base/EnumSelection.h"

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

/**
 * @brief Demangle a cxx11 abi type info to a human readable string.
 *
 * Be aware that this explicitly uses abi::__cxa_demangle and so it might not work for some
 * compilers.
 */
std::string demangle(const char * name);

/**
 * @brief Split a string into tokens given delimiters
 *
 * @param str String to split
 * @param delims Possible delimters
 * @return std::vector<std::string> Tokens split from the given string
 */
std::vector<std::string> split(const std::string & str, const std::string & delims);

/**
 * @brief Trim white space from the two ends of the string
 *
 * @param str String to trim
 * @param white_space Default definition of _whitespace_ characters
 * @return std::string Trimmed string
 */
std::string trim(const std::string & str, const std::string & white_space = " \t\n\v\f\r");

/// Check if the string starts with the given prefix
bool start_with(std::string_view str, std::string_view prefix);

/// Check if the string ends with the given prefix
bool end_with(std::string_view str, std::string_view suffix);

/**
 * @brief Parse a CSV file into a torch::Tensor
 *
 * The first argument \p filename_and_index is the CSV filename. It can optionally provide
 * instructions on how to index the torch::Tensor using the schema
 * `(filename).(extension)[indexing]`.
 *
 * If `indexing` is not provided, the CSV file will be converted to a 2D torch::Tensor, with CSV
 * rows as tensor rows and CSV columns as tensor columns, preserving the order defined in the CSV
 * file.
 *
 * If `indexing` is provided, the 2D torch::Tensor will be indexed accordingly. `indexing`
 * should use the schema `[(row indexing)[,(column indexing)]]` where `row indexing` and `column
 * indexing` are instructions on row indexing and column indexing, respectively. Only single element
 * indexing and the basic slice syntax are supported. See
 * https://numpy.org/doc/stable/user/basics.indexing.html#slicing-and-striding for detailed
 * explanation on the basic slice syntax. Column names can be used in column indexing specification.
 *
 * File name extension
 *
 * Examples of \p filename_and_index specification:
 *
 * The following would convert the _entire_ CSV file to a 2D torch::Tensor
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~python
 * filename.csv
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * A single row can be indexed using
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~python
 * filename.csv[2]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 * A single column can be indexed using
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~python
 * filename.csv[:,3]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 * A slice of row can be indexed using
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~python
 * filename.csv[2:]
 * filename.csv[:5]
 * filename.csv[1:5]
 * filename.csv[:-5]
 * filename.csv[1:5:2]
 * filename.csv[::2]
 * filename.csv[::-1]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 * Row and column indexing can be combined with a comma delimiter.
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~python
 * filename.csv[2:,5:7]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Column names can be used in column indexing (but not in row indexing apparently).
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~python
 * filename.csv[2:,stress_xx:stress_zz]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
torch::Tensor parse_csv(const std::string & filename_and_index,
                        const torch::TensorOptions & options = default_tensor_options());

/**
 * @brief Helper method for parse_csv
 *
 * @return std::tuple<std::string, TorchIndex, TorchIndex> 0: Filename, 1: Row indexing, 2: Column
 * indexing
 */
std::tuple<std::string, TorchIndex, TorchIndex>
parse_csv_spec(const std::string & filename_and_index);

/// Parse a string into TorchIndex
TorchIndex parse_indexing(const std::string & str);

template <typename T>
void
parse_(T & val, const std::string & raw_str)
{
  std::stringstream ss(trim(raw_str));
  ss >> val;
  if (ss.fail() || !ss.eof())
    throw ParserException("Failed to parse '" + raw_str + "' as a " + demangle(typeid(T).name()));
}

template <typename T>
T
parse(const std::string & raw_str)
{
  T val;
  parse_(val, raw_str);
  return val;
}

template <typename T>
void
parse_vector_(std::vector<T> & vals, const std::string & raw_str)
{
  auto tokens = split(raw_str, " \t\n\v\f\r");
  vals.resize(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++)
    parse_<T>(vals[i], tokens[i]);
}

template <typename T>
std::vector<T>
parse_vector(const std::string & raw_str)
{
  std::vector<T> vals;
  parse_vector_(vals, raw_str);
  return vals;
}

template <typename T>
void
parse_vector_vector_(std::vector<std::vector<T>> & vals, const std::string & raw_str)
{
  auto token_vecs = split(raw_str, ";");
  vals.resize(token_vecs.size());
  for (size_t i = 0; i < token_vecs.size(); i++)
    parse_vector_<T>(vals[i], token_vecs[i]);
}

template <typename T>
std::vector<std::vector<T>>
parse_vector_vector(const std::string & raw_str)
{
  std::vector<std::vector<T>> vals;
  parse_vector_vector_(vals, raw_str);
  return vals;
}

// @{ template specializations for parse
template <>
void parse_<bool>(bool &, const std::string & raw_str);
/// This special one is for the evil std::vector<bool>!
template <>
void parse_vector_<bool>(std::vector<bool> &, const std::string & raw_str);
template <>
void parse_<TorchShape>(TorchShape &, const std::string & raw_str);
template <>
void parse_<VariableName>(VariableName &, const std::string & raw_str);
// @}
} // namespace utils
} // namespace neml2
