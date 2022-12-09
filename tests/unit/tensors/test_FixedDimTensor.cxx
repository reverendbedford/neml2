#include <catch2/catch.hpp>

#include "tensors/FixedDimTensor.h"

using namespace neml2;

TEST_CASE("FixedDimTensors have the right shapes, construct blank", "[FixedDimTensors]")
{
  // 2 batch dimensions with sizes (10,2), base dimension (3,4)
  FixedDimTensor<2, 3, 4> A;
  A = A.expand_batch({10, 2});

  SECTION(" batch sizes are correct")
  {
    REQUIRE(A.batch_dim() == 2);
    REQUIRE(A.batch_sizes() == TorchShape({10, 2}));
  }

  SECTION(" base sizes are correct")
  {
    REQUIRE(A.base_dim() == 2);
    REQUIRE(A.base_sizes() == TorchShape({3, 4}));
  }
}

TEST_CASE("FixedDimTensors have the right shapes, construct with tensor", "[FixedDimTensors]")
{
  // 2 batch dimensions with sizes (10,2), base dimension (3,4)
  FixedDimTensor<2, 3, 4> A(torch::zeros({10, 2, 3, 4}, TorchDefaults));

  SECTION(" batch sizes are correct")
  {
    REQUIRE(A.batch_dim() == 2);
    REQUIRE(A.batch_sizes() == TorchShape({10, 2}));
  }

  SECTION(" base sizes are correct")
  {
    REQUIRE(A.base_dim() == 2);
    REQUIRE(A.base_sizes() == TorchShape({3, 4}));
  }
}

TEST_CASE("Not enough required dimensions", "[FixedDimTensor]")
{
#ifndef NDEBUG
  // Can't make this guy, as it won't have enough dimensions for the logical dimensions
  REQUIRE_THROWS(FixedDimTensor<2, 3, 4>(torch::zeros({10, 3, 4}, TorchDefaults)));
#endif
}

TEST_CASE("FixedDimTensors can't be created with the wrong base dimensions", "[FixedDimTensor]")
{
#ifndef NDEBUG
  // Batch is okay, base dimension (5, 4) isn't what we expected (3, 4)
  REQUIRE_THROWS(FixedDimTensor<2, 3, 4>(torch::zeros({10, 2, 5, 4}, TorchDefaults)));
#endif
}
