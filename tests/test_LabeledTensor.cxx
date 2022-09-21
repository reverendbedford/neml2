#include <catch2/catch.hpp>

#include "LabeledTensor.h"

using namespace torch::indexing;

TEST_CASE("Basic functionality for LabeledTensors, including constructing,"
          " labeling a view, and getting/setting data from that view",
          "[LabeledTensor]") {
  // Start with zeros, nbatch = 2
  LabeledTensor<2> A(torch::zeros({10,3,5,4,3}, TorchDefaults));
  
  // Add a couple of views
  A.add_label("first", {Slice(1,3), Slice(), 1});
  A.add_label("second", {0, Slice(1,2), 0});

  SECTION(" shapes come back correctly") {
    REQUIRE(A.get_view("first").sizes() == TorchShape({10,3,2,4}));
    REQUIRE(A.get_view("second").sizes() == TorchShape({10,3,1}));
  }

  SECTION(" can get and set based on views") {
    A.set_view("first", torch::ones({10,3,2,4}));
    REQUIRE(torch::sum(A.get_view("first")).item<double>() == Approx(240));
  }
}

TEST_CASE("All tests work for a LabeledTensor if we provide the views "
          "directly to the constructor", "[LabeledTensor]") {
  // Start with zeros, nbatch = 2
  LabeledTensor<2> A(torch::zeros({10,3,5,4,3}, TorchDefaults), 
                     {
                      {"first", {Slice(1,3), Slice(), 1}},
                      {"second", {0, Slice(1,2), 0}}
                     });
  
  SECTION(" shapes come back correctly") {
    REQUIRE(A.get_view("first").sizes() == TorchShape({10,3,2,4}));
    REQUIRE(A.get_view("second").sizes() == TorchShape({10,3,1}));
  }

  SECTION(" can get and set based on views") {
    A.set_view("first", torch::ones({10,3,2,4}));
    REQUIRE(torch::sum(A.get_view("first")).item<double>() == Approx(240));
  }
}

TEST_CASE("Cannot create two views with the same names",
          "[LabeledTensor]") {
   // Start with zeros, nbatch = 2
  LabeledTensor<2> A(torch::zeros({10,3,5,4,3}, TorchDefaults));
  
  // Add a view
  A.add_label("first", {Slice(1,3), Slice(), 1});

  REQUIRE_THROWS(A.add_label("first", {0, Slice(1,2), 0})); 
}

TEST_CASE("Cannot get a non-existent view",
          "[LabeledTensor]") {
   // Start with zeros, nbatch = 2
  LabeledTensor<2> A(torch::zeros({10,3,5,4,3}, TorchDefaults));
  
  // Add a view
  A.add_label("first", {Slice(1,3), Slice(), 1});

  REQUIRE_THROWS(A.get_view("second")); 
}
