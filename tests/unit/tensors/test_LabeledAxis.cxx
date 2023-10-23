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

#include "neml2/tensors/LabeledAxis.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;

TEST_CASE("LabeledAxis", "[tensors]")
{
  SECTION("class LabeledAxis")
  {
    SECTION("LabeledAxis")
    {
      SECTION("default constructor")
      {
        LabeledAxis a;
        a.setup_layout();
        REQUIRE(a.nitem() == 0);
        REQUIRE(a.nvariable() == 0);
        REQUIRE(a.nsubaxis() == 0);
        REQUIRE(a.storage_size() == 0);
      }

      SECTION("copy constructor")
      {
        LabeledAxis a;
        a.add<Scalar>("scalar");
        a.add<SR2>("r2t");
        a.add<LabeledAxis>("sub");
        a.subaxis("sub").add<Scalar>("scalar");
        a.subaxis("sub").add<SR2>("r2t");

        LabeledAxis b(a);
        b.setup_layout();
        REQUIRE(b.nitem() == 3);
        REQUIRE(b.nvariable() == 2);
        REQUIRE(b.nsubaxis() == 1);
        REQUIRE(b.storage_size() == 14);
      }
    }

    SECTION("add")
    {
      LabeledAxis a;
      LabeledAxisAccessor i({"a", "b", "c"});
      a.add("foo", 13);
      a.add(i, 3);
      a.setup_layout();
      REQUIRE(a.nitem() == 2);
      REQUIRE(a.nvariable() == 1);
      REQUIRE(a.nsubaxis() == 1);
      REQUIRE(a.storage_size() == 16);
    }

    SECTION("rename")
    {
      LabeledAxis a;
      LabeledAxisAccessor i({"a", "b"});
      LabeledAxisAccessor j({"a", "c"});
      a.add(i, 2);
      a.add(j, 3);
      a.rename("a", "d");
      a.subaxis("d").rename("b", "a");
      a.setup_layout();
      REQUIRE(a.has_subaxis("d"));
      REQUIRE(a.subaxis("d").has_variable("a"));
      REQUIRE(a.subaxis("d").has_variable("c"));
    }

    SECTION("remove")
    {
      LabeledAxis a;
      a.add<Scalar>("scalar1");
      a.add<Scalar>("scalar2");
      a.add<Scalar>("scalar3");
      a.add<SR2>("r2t1");
      a.add<SR2>("r2t2");
      a.remove("r2t1");
      a.remove("scalar1");
      a.setup_layout();
      REQUIRE(a.nitem() == 3);
      REQUIRE(a.nvariable() == 3);
      REQUIRE(a.nsubaxis() == 0);
      REQUIRE(a.storage_size() == 8);
      REQUIRE(a.storage_size("scalar2") == 1);
      REQUIRE(a.storage_size("scalar3") == 1);
      REQUIRE(a.storage_size("r2t2") == 6);
    }

    SECTION("clear")
    {
      LabeledAxis a;
      LabeledAxisAccessor i({"a", "b"});
      LabeledAxisAccessor j("c");
      a.add(i, 2);
      a.add(j, 3);
      a.clear();
      a.setup_layout();
      REQUIRE(a.nitem() == 0);
      REQUIRE(a.nvariable() == 0);
      REQUIRE(a.nsubaxis() == 0);
      REQUIRE(a.storage_size() == 0);
    }

    SECTION("merge")
    {
      LabeledAxis a;
      LabeledAxisAccessor i({"a", "b"});
      LabeledAxisAccessor j("c");
      a.add(i, 2);
      a.add(j, 3);

      LabeledAxis b;
      LabeledAxisAccessor k({"a", "d"});
      LabeledAxisAccessor l("c");
      b.add(k, 2);
      b.add(l, 3);

      a.merge(b);

      a.setup_layout();
      b.setup_layout();
      REQUIRE(a.has_variable("c"));
      REQUIRE(a.subaxis("a").has_variable("b"));
      REQUIRE(a.subaxis("a").has_variable("d"));
    }

    SECTION("has_item")
    {
      LabeledAxis a;
      LabeledAxisAccessor i({"a", "b"});
      LabeledAxisAccessor j("c");
      a.add(i, 2);
      a.add(j, 3);
      a.setup_layout();
      REQUIRE(a.has_item("a"));
      REQUIRE(a.has_item("c"));
      REQUIRE(!a.has_item("b"));
    }

    SECTION("has_variable")
    {
      LabeledAxis a;
      LabeledAxisAccessor i({"a", "b"});
      LabeledAxisAccessor j("c");
      a.add(i, 2);
      a.add(j, 3);
      a.setup_layout();
      REQUIRE(!a.has_variable("a"));
      REQUIRE(!a.has_variable("b"));
      REQUIRE(a.has_variable("c"));
      REQUIRE(a.has_variable<Vec>("c"));
      REQUIRE(a.has_variable<Rot>("c"));
      REQUIRE(!a.has_variable<SR2>("c"));
      REQUIRE(a.has_variable(i));
      REQUIRE(a.has_variable(j));
      REQUIRE(!a.has_variable(j.with_suffix("baz")));
    }

    SECTION("storage_size")
    {
      LabeledAxis a;
      LabeledAxisAccessor i({"a", "b"});
      LabeledAxisAccessor j("c");
      LabeledAxisAccessor k({"a", "d"});
      a.add(i, 2);
      a.add(j, 3);
      a.add(k, 3);
      a.setup_layout();
      REQUIRE(a.storage_size() == 8);
      REQUIRE(a.storage_size("a") == 5);
      REQUIRE(a.storage_size("c") == 3);
      REQUIRE(a.storage_size(i) == 2);
      REQUIRE(a.storage_size(j) == 3);
      REQUIRE(a.storage_size(k) == 3);
    }

    SECTION("nested subaxis")
    {
      LabeledAxis a;
      a.add<Scalar>("scalar");
      a.add<SR2>("r2t");
      a.add<LabeledAxis>("sub1");
      a.subaxis("sub1").add<Scalar>("scalar");
      a.subaxis("sub1").add<SR2>("r2t");
      a.subaxis("sub1").add<LabeledAxis>("sub2");
      a.subaxis("sub1").subaxis("sub2").add<Scalar>("scalar");
      a.subaxis("sub1").subaxis("sub2").add<SR2>("r2t");
      a.setup_layout();

      REQUIRE(a.nitem() == 3);
      REQUIRE(a.nvariable() == 2);
      REQUIRE(a.nsubaxis() == 1);
      REQUIRE(a.storage_size() == 21);
      REQUIRE(a.storage_size("scalar") == 1);
      REQUIRE(a.storage_size("r2t") == 6);

      REQUIRE(a.subaxis("sub1").nitem() == 3);
      REQUIRE(a.subaxis("sub1").nvariable() == 2);
      REQUIRE(a.subaxis("sub1").nsubaxis() == 1);
      REQUIRE(a.subaxis("sub1").storage_size() == 14);
      REQUIRE(a.subaxis("sub1").storage_size("scalar") == 1);
      REQUIRE(a.subaxis("sub1").storage_size("r2t") == 6);

      REQUIRE(a.subaxis("sub1").subaxis("sub2").nitem() == 2);
      REQUIRE(a.subaxis("sub1").subaxis("sub2").nvariable() == 2);
      REQUIRE(a.subaxis("sub1").subaxis("sub2").nsubaxis() == 0);
      REQUIRE(a.subaxis("sub1").subaxis("sub2").storage_size() == 7);
      REQUIRE(a.subaxis("sub1").subaxis("sub2").storage_size("scalar") == 1);
      REQUIRE(a.subaxis("sub1").subaxis("sub2").storage_size("r2t") == 6);
    }

    SECTION("chained modifiers")
    {
      LabeledAxis a;
      a.add<Scalar>("scalar1");
      a.add<Scalar>("scalar2");
      a.add<Scalar>("scalar3");
      a.add<SR2>("r2t1");
      a.add<SR2>("r2t2");
      a.add<Scalar>("scalar4").remove("r2t1").rename("scalar4", "scalar5").remove("scalar2");
      a.setup_layout();

      // scalar1
      // scalar3
      // scalar5
      // r2t2

      REQUIRE(a.nitem() == 4);
      REQUIRE(a.nvariable() == 4);
      REQUIRE(a.nsubaxis() == 0);
      REQUIRE(a.storage_size() == 9);
      REQUIRE(a.storage_size("scalar1") == 1);
      REQUIRE(a.storage_size("scalar3") == 1);
      REQUIRE(a.storage_size("scalar5") == 1);
      REQUIRE(a.storage_size("r2t2") == 6);
    }

    SECTION("indices")
    {
      LabeledAxis a;
      a.add<Scalar>("scalar");
      a.add<SR2>("r2t");
      a.add<LabeledAxis>("sub1");
      a.subaxis("sub1").add<Scalar>("scalar");
      a.subaxis("sub1").add<SR2>("r2t");
      a.subaxis("sub1").add<LabeledAxis>("sub2");
      a.subaxis("sub1").subaxis("sub2").add<Scalar>("scalar");
      a.subaxis("sub1").subaxis("sub2").add<SR2>("r2t");
      a.setup_layout();

      /// The sorted layout is
      ///  0 1 2 3 4 5   6          7 8 9 10 11 12   13              14 15 16 17 18 19    20
      /// |----r2t----| |-scalar-| |---sub1/r2t---| |-sub1/scalar-| |--sub1/sub2/r2t--| |-sub1/sub2/scalar-|

      auto idx = torch::arange(a.storage_size());

      REQUIRE(torch::allclose(idx.index(a.indices("r2t")), torch::arange(0, 6)));
      REQUIRE(torch::allclose(idx.index(a.indices("sub1")), torch::arange(7, 21)));

      LabeledAxisAccessor i{{"sub1", "sub2", "r2t"}};
      REQUIRE(torch::allclose(idx.index(a.indices(i)), torch::arange(14, 20)));
    }
  }

  SECTION("operator==")
  {
    // test1 and test2 are equal:
    // They have the same set of items with same names and storage sizes.
    LabeledAxis test1;
    test1.add<Scalar>("scalar1");
    test1.add<SR2>("r2t1");
    test1.add<Scalar>("scalar2");
    test1.add<SR2>("r2t2");
    test1.add<Scalar>("scalar3");
    test1.setup_layout();

    LabeledAxis test2;
    test2.add<SR2>("r2t1");
    test2.add<Scalar>("scalar3");
    test2.add<SR2>("r2t2");
    test2.add<Scalar>("scalar2");
    test2.add<Scalar>("scalar1");
    test2.setup_layout();

    REQUIRE(test1 == test2);
  }

  SECTION("operator!=")
  {
    // test1 and test2 are equal:
    // They have the same set of items with same names and storage sizes.
    LabeledAxis test1;
    test1.add<Scalar>("scalar1");
    test1.add<SR2>("r2t1");
    test1.add<Scalar>("scalar2");
    test1.add<SR2>("r2t2");
    test1.add<Scalar>("scalar3");
    test1.setup_layout();

    LabeledAxis test2;
    test2.add<SR2>("r2t1");
    test2.add<Scalar>("scalar3");
    test2.add<SR2>("r2t2");
    test2.add<Scalar>("scalar2");
    test2.add<Scalar>("scalar1");
    test2.setup_layout();

    // test3 is NOT equal to test1 nor test2
    LabeledAxis test3;
    test3.add<SR2>("r2t1");
    test3.add<Scalar>("scalar3");
    test3.add<SR2>("r2t2");
    test3.setup_layout();

    REQUIRE(test1 != test3);
    REQUIRE(test2 != test3);
  }
}
