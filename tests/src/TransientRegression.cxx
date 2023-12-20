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

#include "TransientRegression.h"
#include "neml2/drivers/TransientDriver.h"
#include <torch/script.h>

namespace fs = std::filesystem;

namespace neml2
{
register_NEML2_object(TransientRegression);

OptionSet
TransientRegression::expected_options()
{
  OptionSet options = Driver::expected_options();
  options.set<std::string>("driver");
  options.set<std::string>("reference");
  options.set<Real>("rtol") = 1e-5;
  options.set<Real>("atol") = 1e-8;
  return options;
}

TransientRegression::TransientRegression(const OptionSet & options)
  : Driver(options),
    _driver(Factory::get_object<TransientDriver>("Drivers", options.get<std::string>("driver"))),
    _reference(options.get<std::string>("reference")),
    _rtol(options.get<Real>("rtol")),
    _atol(options.get<Real>("atol"))
{
  neml_assert(fs::exists(_reference), "Reference file '", _reference.string(), "' does not exist.");
}

bool
TransientRegression::run()
{
  _driver.run();

  // Verify the result
  auto res = torch::jit::load(_driver.save_as_path());
  auto res_ref = torch::jit::load(_reference);
  return allclose(res.named_buffers(true), res_ref.named_buffers(true), _rtol, _atol);
}

bool
allclose(const torch::jit::named_buffer_list & a,
         const torch::jit::named_buffer_list & b,
         Real rtol,
         Real atol)
{
  std::map<std::string, torch::Tensor> a_map;
  for (auto item : a)
    a_map.emplace(item.name, item.value);

  std::map<std::string, torch::Tensor> b_map;
  for (auto item : b)
    b_map.emplace(item.name, item.value);

  for (auto && [key, value] : a_map)
  {
    if (b_map.count(key) == 0)
      return false;
    if (!torch::allclose(value, b_map[key], rtol, atol))
      return false;
  }

  return true;
}
}
