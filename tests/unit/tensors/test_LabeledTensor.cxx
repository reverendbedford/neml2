#include <catch2/catch.hpp>

#include "tensors/LabeledTensor.h"

using namespace torch::indexing;

TEST_CASE("Basic functionality for LabeledTensors", "[LabeledTensor]")
{
  // Start with zeros, nbatch = 2
  LabeledTensor<2> A(torch::zeros({10, 3, 5, 4, 3}, TorchDefaults));

  // Add a couple of views
  A.add_label("first", {Slice(1, 3), Slice(), 1});
  A.add_label("second", {0, Slice(1, 2), 0});

  SECTION(" shapes come back correctly")
  {
    REQUIRE(A["first"].sizes() == TorchShape({10, 3, 2, 4}));
    REQUIRE(A["second"].sizes() == TorchShape({10, 3, 1}));
  }

  SECTION(" can get and set based on views")
  {
    A["first"].index_put_({None}, torch::ones({10, 3, 2, 4}));
    REQUIRE(torch::sum(A["first"]).item<double>() == Approx(240));
  }
}

TEST_CASE("Provide the views directly to the constructor", "[LabeledTensor]")
{
  // Start with zeros, nbatch = 2
  LabeledTensor<2> A(torch::zeros({10, 3, 5, 4, 3}, TorchDefaults),
                     {{"first", {Slice(1, 3), Slice(), 1}}, {"second", {0, Slice(1, 2), 0}}});

  SECTION(" shapes come back correctly")
  {
    REQUIRE(A["first"].sizes() == TorchShape({10, 3, 2, 4}));
    REQUIRE(A["second"].sizes() == TorchShape({10, 3, 1}));
  }

  SECTION(" can get and set based on views")
  {
    A["first"].index_put_({None}, torch::ones({10, 3, 2, 4}));
    REQUIRE(torch::sum(A["first"]).item<double>() == Approx(240));
  }
}

TEST_CASE("Cannot create two views with the same names", "[LabeledTensor]")
{
  // Start with zeros, nbatch = 2
  LabeledTensor<2> A(torch::zeros({10, 3, 5, 4, 3}, TorchDefaults));

  // Add a view
  A.add_label("first", {Slice(1, 3), Slice(), 1});

  REQUIRE_THROWS(A.add_label("first", {0, Slice(1, 2), 0}));
}

TEST_CASE("Cannot get a non-existent view", "[LabeledTensor]")
{
  // Start with zeros, nbatch = 2
  LabeledTensor<2> A(torch::zeros({10, 3, 5, 4, 3}, TorchDefaults));

  // Add a view
  A.add_label("first", {Slice(1, 3), Slice(), 1});

  REQUIRE_THROWS(A["second"]);
}
