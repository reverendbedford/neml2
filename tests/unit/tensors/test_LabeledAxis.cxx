#include <catch2/catch.hpp>

#include "tensors/LabeledAxis.h"

TEST_CASE("Add", "[LabeledAxis]")
{
  // Empty
  LabeledAxis test;

  SECTION("Scalar")
  {
    test.add<Scalar>("scalar");
    test.setup_layout();

    REQUIRE(test.nitem() == 1);
    REQUIRE(test.nvariable() == 1);
    REQUIRE(test.nsubaxis() == 0);
    REQUIRE(test.storage_size() == 1);
    REQUIRE(test.storage_size("scalar") == 1);
  }

  SECTION("SymR2")
  {
    test.add<SymR2>("r2t");
    test.setup_layout();

    REQUIRE(test.nitem() == 1);
    REQUIRE(test.nvariable() == 1);
    REQUIRE(test.nsubaxis() == 0);
    REQUIRE(test.storage_size() == 6);
    REQUIRE(test.storage_size("r2t") == 6);
  }

  SECTION("subaxis")
  {
    test.add<LabeledAxis>("sub");
    test.subaxis("sub").add<Scalar>("scalar");
    test.subaxis("sub").add<SymR2>("r2t");
    test.setup_layout();

    REQUIRE(test.nitem() == 1);
    REQUIRE(test.nvariable() == 0);
    REQUIRE(test.nsubaxis() == 1);
    REQUIRE(test.storage_size() == 7);

    REQUIRE(test.subaxis("sub").nitem() == 2);
    REQUIRE(test.subaxis("sub").nvariable() == 2);
    REQUIRE(test.subaxis("sub").nsubaxis() == 0);
    REQUIRE(test.subaxis("sub").storage_size() == 7);
  }

  SECTION("mixed")
  {
    test.add<Scalar>("scalar1");
    test.add<SymR2>("r2t1");
    test.add<SymR2>("r2t2");
    test.add<Scalar>("scalar2");
    test.add<Scalar>("scalar3");

    test.add<LabeledAxis>("sub1");
    test.subaxis("sub1").add<Scalar>("scalar");
    test.subaxis("sub1").add<SymR2>("r2t");
    test.subaxis("sub1").prefix("sub1");

    test.add<LabeledAxis>("sub2");
    test.subaxis("sub2").add<Scalar>("scalar1");
    test.subaxis("sub2").add<Scalar>("scalar2");
    test.subaxis("sub2").add<SymR2>("r2t");
    test.subaxis("sub2").prefix("sub2");

    test.setup_layout();

    REQUIRE(test.nitem() == 7);
    REQUIRE(test.nvariable() == 5);
    REQUIRE(test.nsubaxis() == 2);
    REQUIRE(test.storage_size() == 30);
    REQUIRE(test.storage_size("scalar1") == 1);
    REQUIRE(test.storage_size("scalar2") == 1);
    REQUIRE(test.storage_size("scalar3") == 1);
    REQUIRE(test.storage_size("r2t1") == 6);
    REQUIRE(test.storage_size("r2t2") == 6);
    REQUIRE(test.storage_size("sub1") == 7);
    REQUIRE(test.storage_size("sub2") == 8);
  }

  SECTION("nested subaxis")
  {
    // State n
    test.add<Scalar>("scalar");
    test.add<SymR2>("r2t");
    test.add<LabeledAxis>("sub1");
    test.subaxis("sub1").add<Scalar>("scalar");
    test.subaxis("sub1").add<SymR2>("r2t");
    test.subaxis("sub1").add<LabeledAxis>("sub2");
    test.subaxis("sub1").subaxis("sub2").add<Scalar>("scalar");
    test.subaxis("sub1").subaxis("sub2").add<SymR2>("r2t");

    test.setup_layout();

    REQUIRE(test.nitem() == 3);
    REQUIRE(test.nvariable() == 2);
    REQUIRE(test.nsubaxis() == 1);
    REQUIRE(test.storage_size() == 21);
    REQUIRE(test.storage_size("scalar") == 1);
    REQUIRE(test.storage_size("r2t") == 6);

    REQUIRE(test.subaxis("sub1").nitem() == 3);
    REQUIRE(test.subaxis("sub1").nvariable() == 2);
    REQUIRE(test.subaxis("sub1").nsubaxis() == 1);
    REQUIRE(test.subaxis("sub1").storage_size() == 14);
    REQUIRE(test.subaxis("sub1").storage_size("scalar") == 1);
    REQUIRE(test.subaxis("sub1").storage_size("r2t") == 6);

    REQUIRE(test.subaxis("sub1").subaxis("sub2").nitem() == 2);
    REQUIRE(test.subaxis("sub1").subaxis("sub2").nvariable() == 2);
    REQUIRE(test.subaxis("sub1").subaxis("sub2").nsubaxis() == 0);
    REQUIRE(test.subaxis("sub1").subaxis("sub2").storage_size() == 7);
    REQUIRE(test.subaxis("sub1").subaxis("sub2").storage_size("scalar") == 1);
    REQUIRE(test.subaxis("sub1").subaxis("sub2").storage_size("r2t") == 6);
  }
}

TEST_CASE("Equality", "[LabeledAxis]")
{
  // test1 and test2 are equal:
  // They have the same set of items with same names and storage sizes.
  LabeledAxis test1;
  test1.add<Scalar>("scalar1");
  test1.add<SymR2>("r2t1");
  test1.add<Scalar>("scalar2");
  test1.add<SymR2>("r2t2");
  test1.add<Scalar>("scalar3");
  test1.setup_layout();

  LabeledAxis test2;
  test2.add<SymR2>("r2t1");
  test2.add<Scalar>("scalar3");
  test2.add<SymR2>("r2t2");
  test2.add<Scalar>("scalar2");
  test2.add<Scalar>("scalar1");
  test2.setup_layout();

  // test3 is NOT equal to test1 nor test2
  LabeledAxis test3;
  test3.add<SymR2>("r2t1");
  test3.add<Scalar>("scalar3");
  test3.add<SymR2>("r2t2");
  test3.setup_layout();

  REQUIRE(test1 == test2);
  REQUIRE(test1 != test3);
  REQUIRE(test2 != test3);
}

TEST_CASE("Miscellaneous modifiers", "[LabeledAxis]")
{
  // Empty
  LabeledAxis test;
  test.add<Scalar>("scalar1");
  test.add<Scalar>("scalar2");
  test.add<Scalar>("scalar3");
  test.add<SymR2>("r2t1");
  test.add<SymR2>("r2t2");

  SECTION("rename")
  {
    test.rename("scalar1", "foo");
    test.setup_layout();

    REQUIRE(test.nitem() == 5);
    REQUIRE(test.nvariable() == 5);
    REQUIRE(test.nsubaxis() == 0);
    REQUIRE(test.storage_size() == 15);
    REQUIRE(test.storage_size("foo") == 1);
    REQUIRE(test.storage_size("scalar2") == 1);
    REQUIRE(test.storage_size("scalar3") == 1);
    REQUIRE(test.storage_size("r2t1") == 6);
    REQUIRE(test.storage_size("r2t2") == 6);
  }

  SECTION("add prefix")
  {
    test.prefix("bar");
    test.setup_layout();

    REQUIRE(test.nitem() == 5);
    REQUIRE(test.nvariable() == 5);
    REQUIRE(test.nsubaxis() == 0);
    REQUIRE(test.storage_size() == 15);
    REQUIRE(test.storage_size("bar_scalar1") == 1);
    REQUIRE(test.storage_size("bar_scalar2") == 1);
    REQUIRE(test.storage_size("bar_scalar3") == 1);
    REQUIRE(test.storage_size("bar_r2t1") == 6);
    REQUIRE(test.storage_size("bar_r2t2") == 6);
  }

  SECTION("add suffix")
  {
    test.suffix("baz");
    test.setup_layout();

    REQUIRE(test.nitem() == 5);
    REQUIRE(test.nvariable() == 5);
    REQUIRE(test.nsubaxis() == 0);
    REQUIRE(test.storage_size() == 15);
    REQUIRE(test.storage_size("scalar1_baz") == 1);
    REQUIRE(test.storage_size("scalar2_baz") == 1);
    REQUIRE(test.storage_size("scalar3_baz") == 1);
    REQUIRE(test.storage_size("r2t1_baz") == 6);
    REQUIRE(test.storage_size("r2t2_baz") == 6);
  }

  SECTION("remove")
  {
    test.remove("r2t1");
    test.remove("scalar1");
    test.setup_layout();

    REQUIRE(test.nitem() == 3);
    REQUIRE(test.nvariable() == 3);
    REQUIRE(test.nsubaxis() == 0);
    REQUIRE(test.storage_size() == 8);
    REQUIRE(test.storage_size("scalar2") == 1);
    REQUIRE(test.storage_size("scalar3") == 1);
    REQUIRE(test.storage_size("r2t2") == 6);
  }

  SECTION("chained modifiers")
  {
    // I'm having fun keeping track of it...
    // foo_scalar1_heh
    // foo_scalar3_heh
    // foo_scalar5_heh
    // foo_r2t2_heh
    test.add<Scalar>("scalar4")
        .remove("r2t1")
        .prefix("foo")
        .rename("scalar4", "scalar5")
        .remove("scalar2")
        .suffix("heh");
    test.setup_layout();

    REQUIRE(test.nitem() == 4);
    REQUIRE(test.nvariable() == 4);
    REQUIRE(test.nsubaxis() == 0);
    REQUIRE(test.storage_size() == 9);
    REQUIRE(test.storage_size("foo_scalar1_heh") == 1);
    REQUIRE(test.storage_size("foo_scalar3_heh") == 1);
    REQUIRE(test.storage_size("foo_scalar5_heh") == 1);
    REQUIRE(test.storage_size("foo_r2t2_heh") == 6);
  }
}
