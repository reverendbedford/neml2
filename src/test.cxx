#include <torch/torch.h>
#include <iostream>

#include "State.h"
#include "BatchedScalar.h"
#include "BatchedSymR2.h"
#include "BatchedSymSymR4.h"
#include "StateInfo.h"

using namespace torch::indexing;

int main()
{
  TorchSize nbatch = 10;

  StateInfo stop;
  stop.add<BatchedScalar>("s1");
  stop.add<BatchedSymR2>("sym1");
  
  StateInfo bottom;
  bottom.add<BatchedSymR2>("sym2");
  bottom.add<BatchedScalar>("s2");
  bottom.add<BatchedScalar>("s3");

  stop.add_substate("substate", bottom);
  
  State state(stop, nbatch);

  std::cout << state.get<BatchedScalar>("s2") << std::endl;
}
