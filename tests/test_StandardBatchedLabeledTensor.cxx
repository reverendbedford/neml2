#include <catch2/catch.hpp>

#include "StandardBatchedLabeledTensor.h"

TEST_CASE("StandardBatchedLabeledTensor has a single batch dimension",
          "[StandardBatchedLabeledTensor]") {
  
  // Batch shape (10,), base shape (3,4)
  StandardBatchedLabeledTensor A(torch::zeros({10,3,4}, TorchDefaults));
  
  SECTION("Correct batch and base shapes") {
    REQUIRE(A.batch_sizes() == TorchShape({10}));
    REQUIRE(A.base_sizes() == TorchShape({3,4}));
  }
}
