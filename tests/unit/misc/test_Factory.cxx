#include <catch2/catch.hpp>

#include "misc/Factory.h"
#include "models/solid_mechanics/NoKinematicHardening.h"
#include "models/solid_mechanics/LinearIsotropicHardening.h"

#include <iostream>

using namespace neml2;

TEST_CASE("Manufacture from an input file", "[Factory]")
{
  std::string fname = "unit/misc/test_Factory.i";
  InputParser parser(fname.c_str());

  // Send the input parameters to the factory to manufacture
  Factory::get_factory().manufacture(parser.root());

  SECTION("model definition")
  {
    const NoKinematicHardening & kinharden = Factory::get_model<NoKinematicHardening>("kinharden");
    REQUIRE(kinharden.input().has_subaxis("state"));
    REQUIRE(kinharden.input().subaxis("state").has_variable<SymR2>("cauchy_stress"));
    REQUIRE(kinharden.output().has_subaxis("state"));
    REQUIRE(kinharden.output().subaxis("state").has_variable<SymR2>("mandel_stress"));

    const LinearIsotropicHardening & isoharden =
        Factory::get_model<LinearIsotropicHardening>("isoharden");
    REQUIRE(isoharden.input().has_subaxis("state"));
    REQUIRE(isoharden.input().subaxis("state").has_variable<Scalar>("equivalent_plastic_strain"));
    REQUIRE(isoharden.output().has_subaxis("state"));
    REQUIRE(isoharden.output().subaxis("state").has_variable<Scalar>("isotropic_hardening"));
  }
}
