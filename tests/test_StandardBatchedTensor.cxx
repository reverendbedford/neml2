#include <catch2/catch.hpp>

#include "StandardBatchedTensor.h"

TEST_CASE("StandardBatchedTensors have the right shapes, construct "
          "blank",
          "[StandardBatchedTensor]")
{

  // Batch dimension 10, base dimension (3,4)
  StandardBatchedTensor<3, 4> A(10);

  SECTION(" batch sizes are correct")
  {
    REQUIRE(A.nbatch() == 1);
    REQUIRE(A.batch_sizes() == TorchShape({10}));
    REQUIRE(A.batch_size() == 10);
  }

  SECTION(" base sizes are correct")
  {
    REQUIRE(A.nbase() == 2);
    REQUIRE(A.base_sizes() == TorchShape({3, 4}));
  }
}

TEST_CASE("StandardBatchedTensors have the right shapes, "
          "construct with tensor",
          "[StandardBatchedTensor]")
{

  // Batch dimension 10, base dimension (3,4)
  StandardBatchedTensor<3, 4> A(torch::zeros({10, 3, 4}, TorchDefaults));

  SECTION(" batch sizes are correct")
  {
    REQUIRE(A.nbatch() == 1);
    REQUIRE(A.batch_sizes() == TorchShape({10}));
    REQUIRE(A.batch_size() == 10);
  }

  SECTION(" base sizes are correct")
  {
    REQUIRE(A.nbase() == 2);
    REQUIRE(A.base_sizes() == TorchShape({3, 4}));
  }
}

TEST_CASE("StandardBatchedTensor can't be created with the wrong base "
          "or batch dimensions",
          "[StandardBatchedTensor]")
{
  REQUIRE_THROWS(StandardBatchedTensor<3, 4>(torch::zeros({10, 5, 4}, TorchDefaults)));
  REQUIRE_THROWS(StandardBatchedTensor<3, 4>(torch::zeros({3, 4}, TorchDefaults)));
}
