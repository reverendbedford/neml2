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

#include <catch2/catch.hpp>

#include "neml2/tensors/R4.h"
#include "neml2/tensors/SymSymR4.h"

using namespace neml2;

TEST_CASE("R4", "[R4]")
{
  SECTION("init from SymSymR4")
  {
    SymSymR4 s(
        torch::tensor({{{0.11470381, 0.17616085, 0.76954039, 0.38373206, 0.42523859, 0.70401029},
                        {0.20102723, 0.93460466, 0.85991842, 0.53950467, 0.70023378, 0.16500279},
                        {0.09899016, 0.74064866, 0.69358815, 0.64908336, 0.47956198, 0.80746042},
                        {0.86170953, 0.48802246, 0.9052055, 0.4934962, 0.50635566, 0.82423019},
                        {0.59510986, 0.03383534, 0.21792362, 0.31903885, 0.17770859, 0.76647519},
                        {0.75508557, 0.05443656, 0.49559057, 0.14082271, 0.42517491, 0.13282324}}},
                      TorchDefaults));
    R4 u = R4::init(s);

    for (TorchSize i = 0; i < 3; i++)
      for (TorchSize j = 0; j < 3; j++)
        for (TorchSize k = 0; k < 3; k++)
          for (TorchSize l = 0; l < 3; l++)
            REQUIRE(torch::allclose(s(i, j, k, l), u(i, j, k, l)));
  }
}
