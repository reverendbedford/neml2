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

#include "utils.h"
#include "neml2/tensors/user_tensors/Orientation.h"

using namespace neml2;

TEST_CASE("Orientation", "[tensors/user_tensors]")
{
  load_model("unit/tensors/user_tensors/test_Orientation.i");

  SECTION("Kocks, radians")
  {
    const auto & auto_1 = Factory::get_object<Rot>("Tensors", "kocks_rad");
    auto correct_1 = Rot(torch::tensor({{-0.98498741, -0.19966683, -9.96664442},
                                        {-1.00250574, -0.02506787, -13.30832395},
                                        {0.0, 0.0, 0.0}},
                                       default_tensor_options()));
    REQUIRE(torch::allclose(auto_1, correct_1));
  }
  SECTION("Kocks, degrees")
  {
    const auto & auto_1 = Factory::get_object<Rot>("Tensors", "kocks_deg");
    auto correct_1 = Rot(torch::tensor({{-0.577350, -1.000000, -1.732051},
                                        {-2.021802, 0.159119, -10.385397},
                                        {0.000000, 0.000000, 0.00000}},
                                       default_tensor_options()));
    REQUIRE(torch::allclose(auto_1, correct_1));
  }

  SECTION("Bunge, radians")
  {
    const auto & auto_1 = Factory::get_object<Rot>("Tensors", "bunge_rad");
    auto correct_1 = Rot(torch::tensor({{-0.101864, 0.010220, -0.202710},
                                        {-0.074953, 0.005632, -0.025005},
                                        {-0.000000, 0.000000, 0.000000}},
                                       default_tensor_options()));
    REQUIRE(torch::allclose(auto_1, correct_1, 1e-3, 1e-4));
  }
  SECTION("Bunge, degrees")
  {
    const auto & auto_1 = Factory::get_object<Rot>("Tensors", "bunge_deg");
    auto correct_1 = Rot(torch::tensor({{-1.000000, 0.577350, -1.732051},
                                        {-0.194084, 0.018688, 0.078702},
                                        {-0.000000, 0.000000, -0.000000}},
                                       default_tensor_options()));
    REQUIRE(torch::allclose(auto_1, correct_1, 1e-3, 1e-4));
  }

  SECTION("Roe, radians")
  {
    const auto & auto_1 = Factory::get_object<Rot>("Tensors", "roe_rad");
    auto correct_1 = Rot(torch::tensor({{-0.010220, -0.101864, -0.202710},
                                        {-0.005632, -0.074953, -0.025005},
                                        {0.000000, 0.000000, 0.000000}},
                                       default_tensor_options()));
    REQUIRE(torch::allclose(auto_1, correct_1, 1e-3, 1e-4));
  }
  SECTION("Roe, degrees")
  {
    const auto & auto_1 = Factory::get_object<Rot>("Tensors", "roe_deg");
    auto correct_1 = Rot(torch::tensor({{-0.577350, -1.000000, -1.732051},
                                        {-0.018688, -0.194084, 0.078702},
                                        {0.000000, -0.000000, -0.000000}},
                                       default_tensor_options()));
    REQUIRE(torch::allclose(auto_1, correct_1, 1e-3, 1e-4));
  }
}
