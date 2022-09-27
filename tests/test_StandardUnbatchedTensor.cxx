#include <catch2/catch.hpp>

#include "StandardUnbatchedTensor.h"

TEST_CASE("StandardUnbatchedTensors have the right shapes, construct "
          "blank",
          "[StandardUnbatchedTensor]")
{

  // No batch dimensions, base dimension (3,4)
  StandardUnbatchedTensor<3, 4> A;

  SECTION(" batch sizes are correct") { REQUIRE(A.nbatch() == 0); }

  SECTION(" base sizes are correct")
  {
    REQUIRE(A.nbase() == 2);
    REQUIRE(A.base_sizes() == TorchShape({3, 4}));
  }
}

TEST_CASE("StandardUnbatchedTensors have the right shapes, "
          "construct with tensor",
          "[StandardUnbatchedTensor]")
{

  // No batch dimensions, base dimension (3,4)
  StandardUnbatchedTensor<3, 4> A(torch::zeros({3, 4}, TorchDefaults));

  SECTION(" batch sizes are correct") { REQUIRE(A.nbatch() == 0); }

  SECTION(" base sizes are correct")
  {
    REQUIRE(A.nbase() == 2);
    REQUIRE(A.base_sizes() == TorchShape({3, 4}));
  }
}

TEST_CASE("StandardUnbatchedTensor can't be created with the wrong base "
          "or batch dimensions",
          "[StandardUnbatchedTensor]")
{
  REQUIRE_THROWS(StandardUnbatchedTensor<3, 4>(torch::zeros({5, 4}, TorchDefaults)));
  REQUIRE_THROWS(StandardUnbatchedTensor<3, 4>(torch::zeros({10, 3, 4}, TorchDefaults)));
}
