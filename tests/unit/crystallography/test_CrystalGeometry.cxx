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

#include <filesystem>

#include "utils.h"
#include "neml2/base/Factory.h"
#include "neml2/models/crystallography/CrystalGeometry.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;
using namespace neml2::crystallography;
namespace fs = std::filesystem;

TEST_CASE("CrystalGeometry", "[crystallography]")
{
  torch::manual_seed(42);
  const auto & DTO = default_tensor_options;

  // Load all the models in
  Factory::clear();
  load_model(fs::absolute("unit/crystallography/test_CrystalGeometry.i"));

  SECTION("Simple cubic model")
  {
    auto & model = Factory::get_object<CrystalGeometry>("Models", "scgeom");

    SECTION("Lattice vectors are okay")
    {
      REQUIRE(torch::allclose(model.a1(), Vec::fill(1.2, 0.0, 0.0, DTO)));
      REQUIRE(torch::allclose(model.a2(), Vec::fill(0.0, 1.2, 0.0, DTO)));
      REQUIRE(torch::allclose(model.a3(), Vec::fill(0.0, 0.0, 1.2, DTO)));
    }

    SECTION("Reciprocol lattice vectors are okay")
    {
      REQUIRE(torch::allclose(model.b1(), Vec::fill(1.0 / 1.2, 0.0, 0.0, DTO)));
      REQUIRE(torch::allclose(model.b2(), Vec::fill(0.0, 1.0 / 1.2, 0.0, DTO)));
      REQUIRE(torch::allclose(model.b3(), Vec::fill(0.0, 0.0, 1.0 / 1.2, DTO)));
    }
  }
}