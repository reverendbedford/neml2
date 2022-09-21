#include <catch2/catch.hpp>

#include "SymSymR4.h"
#include "SymR2.h"
#include "ElasticityTensors.h"

using namespace torch::indexing;

TEST_CASE("SymSymR4 dotted with SymR2", "[SymSymR4]") {
  // I'm suffering from a lack of setup functions
  SymSymR4 C = fill_isotropic(100.0, 0.3);

  SECTION("SymR2") {
    SymR2 other(torch::tensor({1.0,2.0,3.0,4.0,5.0,6.0}, TorchDefaults));
    SymR2 result = C.dot(other);
    // Separate calculation in numpy
    SymR2 correct = SymR2(torch::tensor(
            {423.07692308, 500.0, 576.92307692, 307.69230769,384.61538462,461.53846154},
            TorchDefaults));
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("BatchedSymR2") {
    int nbatch = 10;
    BatchedSymR2 other(torch::repeat_interleave(
            torch::tensor({1.0,2.0,3.0,4.0,5.0,6.0}, TorchDefaults).reshape(
                {1,6}), nbatch, 0));
    BatchedSymR2 result = C.dot(other);
    BatchedSymR2 correct = BatchedSymR2(torch::repeat_interleave(torch::tensor(
            {423.07692308, 500.0, 576.92307692, 307.69230769,384.61538462,461.53846154},
            TorchDefaults).reshape({1,6}), nbatch, 0));
    REQUIRE(torch::allclose(result, correct));

  }
}
