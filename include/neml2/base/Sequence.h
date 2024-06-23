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

#include <vector>
#include <string>

#include "neml2/misc/parser_utils.h"
#include "neml2/contrib/csv.hpp"

namespace neml2
{
// Forward decl
template <typename T>
class Sequence;
template <typename T>
std::stringstream & operator>>(std::stringstream &, Sequence<T> &);

/**
 * @brief A lazily parsed sequence of data
 *
 * This class is primarily designed to help parse CSV files, similar to neml2::CrossRef.
 * Upon constuction, this object holds a single string without any further parser. The string can be
 * parsed in one of two ways:
 *   1. As a space delimited vector of data of type @tparam T. In this mode, the string is simply
 *      parsed using utils::parse<std::vector<T>>.
 *   2. As a column in a csv file. In this mode, the string must conform to the schema
 *      "path/to/filename.csv:<column-name|[column-index]>", where the part before the colon
 *      represents the CSV file path, and the part after the colon represents the column identifier.
 *      The column identifier can either be the column name or the column index. The column index
 *      should be enclosed by square brackets.
 *
 * One apparent caveat is that if the underlying string is not space delimited and has the pattern
 * "*.csv:*", it cannot be parsed into a std::vector<std::string>.
 *
 * @tparam T
 */
template <typename T>
class Sequence
{
public:
  Sequence() = default;

  /**
   * @brief Assignment operator
   *
   * This simply assigns the string without further parsing
   */
  Sequence<T> & operator=(const std::string & other);

  /// Implicit conversion operator.
  operator std::vector<T>() const;

  /// Convert to a vector
  std::vector<T> vec() const;

  /// Test equality
  bool operator==(const Sequence<T> & other) const { return _raw_str == other.raw(); }

  /// Get the raw string literal
  const std::string & raw() const { return _raw_str; }

  friend std::stringstream & operator>> <>(std::stringstream & in, Sequence<T> &);

private:
  /// The raw string literal.
  std::string _raw_str;
};
} // namespace neml2

///////////////////////////////////////////////////////////////////////////////
// Implementations
///////////////////////////////////////////////////////////////////////////////

namespace neml2
{
template <typename T>
Sequence<T> &
Sequence<T>::operator=(const std::string & other)
{
  _raw_str = other;
  return *this;
}

template <typename T>
Sequence<T>::operator std::vector<T>() const
{
  return vec();
}

template <typename T>
std::vector<T>
Sequence<T>::vec() const
{
  // If the string is empty, just return an empty vector
  if (_raw_str.empty())
    return std::vector<T>();

  // Parse into a vector of string, and if the vector has length greater than one, then this
  // must be a vector instead of a CSV file column.
  auto str_vec = utils::parse_vector<std::string>(_raw_str);
  if (str_vec.size() > 1)
  {
    std::vector<T> v;
    for (const auto & s : str_vec)
      v.push_back(utils::parse<T>(s));
    return v;
  }

  // If the string vector has exactly one element, we need to check if it conforms with our CSV
  // file column schema

  // Case 1: It doesn't conform the the CSV column schema, then just parse it normally
  if (str_vec[0].find(".csv:") == std::string::npos)
    return {utils::parse<T>(str_vec[0])};

  // Case 2: Invoke the 3rd party csv reader
  // Remove white space
  const auto str_trimmed = utils::trim(_raw_str);

  // Find file name and column identifier
  const auto pos = str_trimmed.find(".csv:");
  const auto filename = str_trimmed.substr(0, pos + 4);
  const auto col_id = str_trimmed.substr(pos + 5);

  std::vector<T> v;
  try
  {
    // Open the CSV
    csv::CSVReader reader(filename);

    // Get the column index
    int col;
    if (col_id.front() == '[' && col_id.back() == ']')
      col = utils::parse<int>(col_id.substr(1, col_id.length() - 2));
    else
      col = reader.index_of(col_id);

    // Loop through rows to fill the vector
    for (const auto & row : reader)
      v.push_back(utils::parse<T>(row[col].get<std::string>()));
  }
  catch (std::exception & e)
  {
    throw NEMLException("The following error was thrown while reading the CSV column '" + _raw_str +
                        "':\n" + e.what());
  }

  return v;
}

template <typename T>
std::ostream &
operator<<(std::ostream & os, const Sequence<T> & seq)
{
  os << seq.raw();
  return os;
}

template <typename T>
std::stringstream &
operator>>(std::stringstream & ss, Sequence<T> & seq)
{
  // This is special -- we have to eat the entire stream
  seq._raw_str = ss.str();
  ss.setstate(std::ios_base::eofbit);
  return ss;
}
} // namespace neml2
