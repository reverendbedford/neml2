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

#include "StructuralDriver.h"
#include "neml2/misc/math.h"
#include "neml2/base/HITParser.h"

#include "VerificationTest.h"

using namespace neml2;

TEST_CASE("Perzyna viscoplasticity verification tests", "[StructuralVerificationTests][Perzyna]")
{
  HITParser parser;
  parser.parse_and_manufacture("verification/solid_mechanics/perzyna/isolinear.i");
  auto & model = Factory::get_object<Model>("Models", "model");

  SECTION("Linear isotropic hardening, uniaxial")
  {
    // Load and run the test
    std::string fname = "verification/solid_mechanics/perzyna/isolinear_uniaxial.vtest";
    VerificationTest test(fname);
    torch::NoGradGuard no_grad_guard;
    REQUIRE(test.compare(model));
  }

  SECTION("Linear isotropic hardening, multiaxial")
  {
    // Load and run the test
    std::string fname = "verification/solid_mechanics/perzyna/isolinear_multiaxial.vtest";
    VerificationTest test(fname);
    torch::NoGradGuard no_grad_guard;
    REQUIRE(test.compare(model));
  }
}

TEST_CASE("Perzyna viscoplasticity with Voce hardening", "[StructuralVerificationTests]")
{
  HITParser parser;
  parser.parse_and_manufacture("verification/solid_mechanics/perzyna/voce.i");
  auto & model = Factory::get_object<Model>("Models", "model");

  SECTION("Voce isotropic hardening, uniaxial")
  {
    // Load and run the test
    std::string fname = "verification/solid_mechanics/perzyna/voce.vtest";
    VerificationTest test(fname);
    torch::NoGradGuard no_grad_guard;
    REQUIRE(test.compare(model));
  }
}

TEST_CASE("Perzyna viscoplasticity with combined hardening", "[StructuralVerificationTests]")
{
  HITParser parser;
  parser.parse_and_manufacture("verification/solid_mechanics/perzyna/combined.i");
  auto & model = Factory::get_object<Model>("Models", "model");

  SECTION("Uniaxial load")
  {
    // Load and run the test
    std::string fname = "verification/solid_mechanics/perzyna/combined.vtest";
    VerificationTest test(fname);
    torch::NoGradGuard no_grad_guard;
    REQUIRE(test.compare(model));
  }
}
