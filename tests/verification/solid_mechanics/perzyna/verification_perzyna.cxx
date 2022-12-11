#include <catch2/catch.hpp>

#include "VerificationTest.h"

TEST_CASE("Perzyna viscoplasticity verification tests",
          "[StructuralVerificationTests]")
{
  SECTION("Linear isotropic hardening, uniaxial")
  {
    std::string fname = "verification/solid_mechanics/perzyna/isolinear_uniaxial.vtest";
    VerificationTest test(fname);

    std::cout << test.strain() << std::endl;
  }
}
