#include <catch2/catch.hpp>

#include "WorkDispatcher.h"
#include "SmallStrainIsotropicLinearElasticModel.h"

TEST_CASE("chunked_map", "[WorkDispatcher]")
{
  using namespace torch::indexing;
  SmallStrainIsotropicLinearElasticModel model(100, 0.3);

  TorchSize nbatch = 1024 * 10;
  TorchSize chunk_size = 1024;
  Progress progress(nbatch);

  // Fairly arbitrary strains and previous stress...
  SymR2 strain_np1(torch::tensor({0.25, 0.0, -0.05, 0.15, 0.1, -0.5}, TorchDefaults), nbatch);
  SymR2 strain_n(torch::tensor({0.1, 0.05, 0.5, -0.075, 0.7, -0.1}, TorchDefaults), nbatch);
  SymR2 stress_n(torch::tensor({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, TorchDefaults), nbatch);
  Scalar time_np1(1, nbatch);
  Scalar time_n(0, nbatch);

  // Actual state objects
  State state_n(model.state(), stress_n);

  State forces_np1(model.forces(), nbatch);
  forces_np1.set<SymR2>("strain", strain_np1);
  forces_np1.set<Scalar>("time", time_np1);

  State forces_n(model.forces(), nbatch);
  forces_n.set<SymR2>("strain", strain_n);
  forces_n.set<Scalar>("time", time_n);

  std::vector<State> state_np1_chunks;
  chunked_map(
      [&](TorchSize begin, TorchSize end)
      {
        const auto range = Slice(begin, end);
        State forces_np1_chunk(forces_np1.info(), forces_np1.tensor().batch_index({range}));
        State state_n_chunk(state_n.info(), state_n.tensor().batch_index({range}));
        State forces_n_chunk(forces_n.info(), forces_n.tensor().batch_index({range}));
        State state_np1_chunk = model.state_update(forces_np1_chunk, state_n_chunk, forces_n_chunk);
        state_np1_chunks.push_back(state_np1_chunk);
      },
      chunk_size,
      progress);
}
