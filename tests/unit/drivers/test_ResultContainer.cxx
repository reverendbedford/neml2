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

#include "neml2/drivers/ResultContainer.h"

using namespace neml2;

TEST_CASE("ResultContainer", "[ResultContainer]")
{
  TorchSize nbatch = 10;
  LabeledAxis info;
  info.add<SymR2>("first").add<Scalar>("second").add<Scalar>("third");
  info.setup_layout();
  LabeledVector A(nbatch, {&info});
  A.set(torch::ones({nbatch, 6}), "first");
  LabeledVector B(nbatch, {&info});
  B.set(torch::ones({nbatch, 6}) * 3, "first");
  ResultContainer rc;
  rc.emplace("A", A);
  rc.emplace("B", A);
  rc.emplace("B", B);

  REQUIRE(math::allclose(rc, rc));

  SECTION("Keys do not match")
  {
    ResultContainer rc2;
    rc2.emplace("C", A);
    REQUIRE(!math::allclose(rc, rc2));
  }

  SECTION("Values do not match")
  {
    ResultContainer rc2;
    rc2.emplace("A", A);
    rc2.emplace("B", A);
    REQUIRE(!math::allclose(rc, rc2));
  }
}
