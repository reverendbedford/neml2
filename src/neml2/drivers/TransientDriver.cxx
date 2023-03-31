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

#include "neml2/drivers/TransientDriver.h"

namespace fs = std::filesystem;

namespace neml2
{
ParameterSet
TransientDriver::expected_params()
{
  ParameterSet params = Driver::expected_params();
  params.set<std::string>("save_as");
  return params;
}

TransientDriver::TransientDriver(const ParameterSet & params)
  : Driver(params),
    _save_as(params.get<std::string>("save_as")),
    _result(std::make_shared<ResultSeriesContainer>())
{
}

bool
TransientDriver::run()
{
  auto status = solve();

  if (!save_as_path().empty())
    output();

  return status;
}

std::string
TransientDriver::save_as_path() const
{
  return _save_as;
}

void
TransientDriver::output() const
{
  if (_verbose)
    std::cout << "Saving results..." << std::endl;

  auto cwd = fs::current_path();
  auto out = cwd / save_as_path();

  if (out.extension() == ".pt")
    output_pt(out);
  else
    neml_assert(false, "Unsupported output format: ", out.extension());

  if (_verbose)
    std::cout << "Results saved to " << save_as_path() << std::endl;
}

void
TransientDriver::output_pt(const std::filesystem::path & out) const
{
  torch::save(result(), out);
}
} // namespace neml2
