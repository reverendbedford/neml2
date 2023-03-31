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

ParameterSet
TransientRegression::expected_params()
{
  ParameterSet params = Driver::expected_params();
  params.set<std::string>("driver");
  params.set<std::string>("reference");
  params.set<Real>("rtol") = 1e-5;
  params.set<Real>("atol") = 1e-8;
  return params;
}

TransientRegression::TransientRegression(const ParameterSet & params)
  : Driver(params),
    _driver(Factory::get_object<TransientDriver>("Drivers", params.get<std::string>("driver"))),
    _reference(params.get<std::string>("reference")),
    _rtol(params.get<Real>("rtol")),
    _atol(params.get<Real>("atol"))
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
  return utils::allclose(res.named_buffers(true), res_ref.named_buffers(true));
}
}
