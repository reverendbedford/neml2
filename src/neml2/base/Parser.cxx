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

#include "neml2/base/Parser.h"
#include "neml2/base/Factory.h"
#include "neml2/base/HITParser.h"

namespace neml2
{
std::vector<std::string> Parser::sections = {"Tensors", "Solvers", "Data", "Models", "Drivers"};

void
load_model(const std::string & path, const std::string & additional_input, ParserType ptype)
{
  // We are being forward looking here
  if (ptype == ParserType::AUTO)
  {
    if (utils::end_with(path, ".i"))
      ptype = ParserType::HIT;
    else if (utils::end_with(path, ".xml"))
      ptype = ParserType::XML;
    else if (utils::end_with(path, ".yml"))
      ptype = ParserType::YAML;
  }

  // but for now we only support HIT
  if (ptype == ParserType::HIT)
  {
    HITParser parser;

    Factory::clear();
    Factory::load(parser.parse(path, additional_input));
  }
  else
    neml_assert(false, "Unsupported parser type");
}
} // namespace neml2
