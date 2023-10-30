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
#include "neml2/drivers/Driver.h"

using namespace neml2;
namespace fs = std::filesystem;

TEST_CASE("models")
{
  const auto pwd = fs::current_path();
  const auto search_path = fs::path(NEML2_BINARY_DIR) / "tests" / "unit" / "models";
  const auto tests_path = fs::path(NEML2_SOURCE_DIR) / "tests" / "unit" / "models";

  // Find all unit tests
  std::vector<fs::path> tests;
  using rdi = fs::recursive_directory_iterator;
  for (const auto & entry : rdi(search_path))
    if (entry.path().extension() == ".i")
      tests.push_back(fs::absolute(entry.path()));

  for (auto test : tests)
  {
    // Change current working directory to the parent directory of the input file
    fs::current_path(test.parent_path());
    const auto cwd = fs::current_path();

    // When we do development, only symbolic links are created for test artifacts.
    // Otherwise, the tests artifacts are copied over.
    auto section_name = fs::is_symlink(test) ? fs::relative(test, tests_path).string()
                                             : fs::relative(test, search_path).string();

    DYNAMIC_SECTION(section_name)
    {
      try
      {
        // Load and run the model
        load_model(test.filename());
        auto & driver = Factory::get_object<Driver>("Drivers", "unit");
        REQUIRE(driver.run());
      }
      catch (...)
      {
        // See the comments below. Let's go back to where we were before completely giving up...
        fs::current_path(pwd);
        throw;
      }
    }
  }

  // Catch2 will split dynamic sections into different test cases, so we need to set the current
  // path back to where we were. Otherwise the next test case will start from the current path
  fs::current_path(pwd);
}
