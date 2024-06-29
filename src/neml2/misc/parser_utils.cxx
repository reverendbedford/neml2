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
#include "csvparser/csv.hpp"

#include <cxxabi.h>
#include <regex>

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

BatchTensor
parse_csv(const std::string & csv, const torch::TensorOptions & options)
{
  try
  {
    // Separate filename.extension[indexing] and (batch-shape)
    std::string filename_and_index;
    TorchShape batch_shape;
    const bool has_batch_shape = csv.back() == ')';
    if (has_batch_shape)
    {
      auto pos = csv.find_last_of("(");
      if (pos == std::string::npos)
        throw ParserException("Missing starting ( in '" + csv + "'");
      filename_and_index = csv.substr(0, pos);
      batch_shape = parse<TorchShape>(csv.substr(pos));
    }
    else
      filename_and_index = csv;

    // Separate filename, row indexing, and column indexing
    auto [filename, rows, cols] = parse_csv_spec(filename_and_index);

    // Allow the following delimiters
    csv::CSVFormat fmt;
    fmt.delimiter({',', ' ', '\t'});

    // Count number of rows and columns in the CSV file
    csv::CSVReader counter(filename, fmt);
    TorchSize nrow = 0;
    for (const auto & row : counter)
    {
      (void)row;
      nrow += 1;
    }
    auto ncol = counter.get_col_names().size();

    // Figure out which rows to read
    auto row_indices = torch::full({nrow}, false, torch::kBool);
    row_indices.index_put_(rows, true);
    std::vector<bool> row_indices_v(row_indices.data_ptr<bool>(),
                                    row_indices.data_ptr<bool>() + row_indices.numel());

    // Figure out which columns to read
    auto col_indices = torch::arange(TorchSize(ncol)).index(cols).contiguous();
    std::vector<TorchSize> col_indices_v(col_indices.data_ptr<TorchSize>(),
                                         col_indices.data_ptr<TorchSize>() + col_indices.numel());

    // Read the CSV
    csv::CSVReader reader(filename, fmt);
    auto tensor = torch::empty(
        {torch::sum(row_indices).item<TorchSize>(), TorchSize(col_indices_v.size())}, options);
    TorchSize row_number = 0;
    TorchSize i = 0;
    for (const auto & row : reader)
      if (row_indices_v[row_number++])
      {
        std::vector<Real> row_v;
        for (auto col_index : col_indices_v)
          row_v.push_back(row[col_index].get<Real>());
        tensor.index_put_({i++}, torch::tensor(row_v, options));
      }

    // Convert torch::Tensor to neml2::BatchTensor
    TorchSize batch_dim = i == 1 ? 0 : 1;
    auto batch_tensor = BatchTensor(tensor.squeeze(), batch_dim);
    return has_batch_shape ? batch_tensor.batch_reshape(batch_shape) : batch_tensor;
  }
  catch (const ParserException & e)
  {
    throw;
  }
  catch (const NEMLException & e)
  {
    throw;
  }
  catch (const std::exception & e)
  {
    throw ParserException("Encountered an unhandled error while reading CSV '" + csv +
                          "': " + e.what());
  }
}

std::tuple<std::string, TorchIndex, TorchIndex>
parse_csv_spec(const std::string & filename_and_index)
{
  size_t pos;
  std::string filename, index;

  // Check to see if indexing is specified
  pos = filename_and_index.find('[');
  if (filename_and_index.back() == ']' && pos != std::string::npos)
  {
    // Split filename and index
    filename = filename_and_index.substr(0, pos);
    index = filename_and_index.substr(pos);
  }
  else
    filename = filename_and_index;

  // Allow the following delimiters
  csv::CSVFormat fmt;
  fmt.delimiter({',', ' ', '\t'});
  csv::CSVReader reader(filename, fmt);
  auto col_names = reader.get_col_names();
  while (col_names[0] == "#")
  {
    fmt.h reader = csv::CSVReader(filename, fmt);
  }

  // Read the CSV

  for (auto col_name : col_names)
    std::cout << col_name << " ";
  std::cout << std::endl;

  // Parse row and col indexing
  auto rows = TorchIndex(torch::indexing::Slice());
  auto cols = TorchIndex(torch::indexing::Slice());
  if (!index.empty())
  {
    // Remove the enclosing brackets
    index = index.substr(1, index.length() - 2);
    if (index.empty())
      throw ParserException("Empty CSV indexing not allowed");
    // Split row and col index
    const bool adv_row_idx = (index[0] == '[');
    if (adv_row_idx)
    {
      pos = index.find(']');
      if (pos == std::string::npos)
        throw ParserException("Missing closing ] in '" + filename_and_index + "'");
      auto row_idx = index.substr(0, pos + 1);
      rows = parse_indexing(row_idx);
      if (pos + 2 < index.length())
        if (index[pos + 1] != ',')
          throw ParserException("Expected comma after row indexing, got '" +
                                stringify(index[pos + 1]) + "' instead");
      pos += 1;
    }
    else
    {
      pos = index.find(',');
      if (pos == 0)
        throw ParserException("Row indexing cannot begin with comma");
      auto row_idx = index.substr(0, pos);
      rows = parse_indexing(row_idx);
    }
    if (pos != std::string::npos && pos + 1 < index.length())
    {
      auto col_idx = index.substr(pos + 1);
      if (!col_idx.empty())
      {
        // Substitute column names for integer
        for (const auto & col_name : col_names)
          col_idx = std::regex_replace(
              col_idx, std::regex(col_name), stringify(reader.index_of(col_name)));
        cols = parse_indexing(col_idx);
      }
    }
  }

  return {filename, rows, cols};
}

TorchIndex
parse_indexing(const std::string & str)
{
  size_t pos;

  // Advanced indexing
  if (str[0] == '[')
  {
    if (str.back() != ']')
      throw ParserException("Missing closing ] in '" + str + "'");
    auto indices_str = split(str.substr(1, str.length() - 2), ",");
    std::vector<TorchSize> indices;
    for (const auto & index_str : indices_str)
      indices.push_back(parse<TorchSize>(index_str));
    return torch::tensor(indices, default_integer_tensor_options());
  }

  // Single element indexing
  if (str.find(':') == std::string::npos)
    return parse<Integer>(str);

  // Slice
  std::string start, stop, step;
  pos = str.find(':');
  start = str.substr(0, pos);
  if (pos != std::string::npos && pos + 1 < str.length())
  {
    auto stop_step = str.substr(pos + 1);
    pos = stop_step.find(':');
    stop = stop_step.substr(0, pos);
    if (pos != std::string::npos && pos + 1 < stop_step.length())
      step = stop_step.substr(pos + 1);
  }
  return torch::indexing::Slice(
      start.empty() ? torch::nullopt : std::optional(torch::SymInt(parse<TorchSize>(start))),
      stop.empty() ? torch::nullopt : std::optional(torch::SymInt(parse<TorchSize>(stop))),
      step.empty() ? torch::nullopt : std::optional(torch::SymInt(parse<TorchSize>(step))));
}

template <>
void
parse_<bool>(bool & val, const std::string & raw_str)
{
  std::string str_val = parse<std::string>(raw_str);
  if (str_val == "true")
    val = true;
  else if (str_val == "false")
    val = false;
  else
    throw ParserException("Failed to parse boolean value. Only 'true' and 'false' are recognized.");
}

template <>
void
parse_vector_(std::vector<bool> & vals, const std::string & raw_str)
{
  auto tokens = split(raw_str, " \t\n\v\f\r");
  vals.resize(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++)
    vals[i] = parse<bool>(tokens[i]);
}

template <>
void
parse_<VariableName>(VariableName & val, const std::string & raw_str)
{
  auto tokens = split(raw_str, "/ \t\n\v\f\r");
  val = VariableName(tokens);
}

template <>
void
parse_<TorchShape>(TorchShape & val, const std::string & raw_str)
{
  if (!start_with(raw_str, "(") || !end_with(raw_str, ")"))
    throw ParserException("Trying to parse " + raw_str +
                          " as a shape, but a shape must start with '(' and end with ')'");

  auto inner = trim(raw_str, "() \t\n\v\f\r");
  auto tokens = split(inner, ", \t\n\v\f\r");

  val.clear();
  for (auto & token : tokens)
    val.push_back(parse<TorchSize>(token));
}
} // namespace utils
} // namespace neml2
