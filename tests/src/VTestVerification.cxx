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

#include "VTestVerification.h"
#include "neml2/drivers/TransientDriver.h"

namespace neml2
{
register_NEML2_object(VTestVerification);

OptionSet
VTestVerification::expected_options()
{
  OptionSet options = Driver::expected_options();
  options.set<std::string>("driver");
  options.set<std::vector<std::string>>("variables");
  options.set<std::vector<CrossRef<torch::Tensor>>>("references");
  options.set<Real>("rtol") = 1e-5;
  options.set<Real>("atol") = 1e-8;
  options.set<bool>("taylor_average") = false;
  return options;
}

VTestVerification::VTestVerification(const OptionSet & options)
  : Driver(options),
    _driver(Factory::get_object<TransientDriver>("Drivers", options.get<std::string>("driver"))),
    _variables(options.get<std::vector<std::string>>("variables")),
    _references(options.get<std::vector<CrossRef<torch::Tensor>>>("references")),
    _rtol(options.get<Real>("rtol")),
    _atol(options.get<Real>("atol")),
    _taylor_average(options.get<bool>("taylor_average"))
{
  neml_assert(_variables.size() == _references.size(),
              "Must provide the same number of variables and reference variables. ",
              _variables.size(),
              " variables provided, while ",
              _references.size(),
              " reference variables provided.");
}

bool
VTestVerification::run()
{
  _driver.run();

  // Verify the variable values against the references
  for (size_t i = 0; i < _variables.size(); i++)
    if (!allclose(_variables[i], _references[i]))
      return false;

  return true;
}

bool
VTestVerification::allclose(const std::string & var, torch::Tensor ref) const
{
  // NEML2 results
  const auto res = _driver.result();

  // Variable to check
  auto tokens = utils::split(var, ".");
  auto axis = tokens[0];
  auto name = tokens[1];
  auto neml2_tensor = res->at<torch::nn::Module>(axis).named_buffers(true)[name];

  // Average, if requested
  torch::Tensor comp;
  if (_taylor_average)
    comp = torch::sum(neml2_tensor, 1).unsqueeze(1) / ((Real)neml2_tensor.sizes()[1]);
  else
    comp = neml2_tensor;

  // Check
  if (!torch::allclose(comp, ref, _rtol, _atol))
    return false;

  return true;
}
}
