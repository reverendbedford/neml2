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

#include "neml2/base/Factory.h"

namespace neml2
{
/**
 * @brief A parser is responsible for parsing an input file into a collection of parameters which
 * can be used by the `Factory` to manufacture corresponding objects.
 *
 */
class Parser
{
public:
  Parser() = default;

  /**
   * @brief Deserialize a file and manufacture all objects.
   *
   * @param filename Name/path of the input file.
   * @param additional_input  Additional content of the input file not included in the input file
   * itself, e.g., from command line.
   */
  virtual void parse_and_manufacture(const std::string & filename,
                                     const std::string & additional_input = "");

  /**
   * @brief Deserialize a file.
   *
   * @param filename Name/path of the input file.
   * @param additional_input  Additional content of the input file not included in the input file
   * itself, e.g., from command line.
   * @return ParameterCollection The extracted object parameters.
   */
  virtual ParameterCollection parse(const std::string & filename,
                                    const std::string & additional_input = "") const = 0;
};

} // namespace neml2
