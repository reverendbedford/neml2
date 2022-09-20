#include <catch2/catch.hpp>

#include "ElasticityTensors.h"

TEST_CASE("Correctly fill isotropic tensor",
          "[ElasticityTensors]") {
  Scalar E = 100.0;
  Scalar nu = 0.3;

  SECTION("values in the correct places") {
    SymSymR4 C = fill_isotropic(E, nu);
    double Ev = E.value();
    double nuv = nu.value();
    double pf = Ev/((1+nuv)*(1-2*nuv));
    double C1 = (1-nuv) * pf;
    double C2 = nuv * pf;
    double C4 = (1-2*nuv) * pf;
    double z = 0.0;
    torch::Tensor right = torch::tensor(
        {
        {C1,C2,C2,z, z, z },
        {C2,C1,C2,z, z, z },
        {C2,C2,C1,z, z, z },
        {z, z, z, C4,z, z },
        {z, z, z, z, C4,z },
        {z, z, z, z, z, C4}
        });
    REQUIRE(torch::allclose(C, right));
  }
}
