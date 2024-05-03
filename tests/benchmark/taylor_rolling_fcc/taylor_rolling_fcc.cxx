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

#include <catch2/catch_test_macros.hpp>

#include "utils.h"
#include "neml2/drivers/Driver.h"

using namespace neml2;

TEST_CASE("taylor")
{
  std::vector<TorchSize> nbatches = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
  std::vector<std::string> devices = {"cpu", "cuda:0"};
  TorchSize ntime = 500;

  for (auto && device : devices)
    for (TorchSize nbatch : nbatches)
    {
      const auto config = "nbatch=" + utils::stringify(nbatch) + " device=" + device +
                          " ntime=" + utils::stringify(ntime);
      load_model("benchmark/taylor_rolling_fcc/model.i", config);
      auto & driver = Factory::get_object<Driver>("Drivers", "driver");
      BENCHMARK("{" + config + "}") { return driver.run(); };
    }
}
