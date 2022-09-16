#include <torch/torch.h>
#include <iostream>

#include "State.h"
#include "StateDerivative.h"
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

  stop.add_substate("another", bottom);
  
  State state(stop, nbatch);

  std::cout << state.get<BatchedScalar>("s2") << std::endl;

  State sub = state.get_substate("another");

  std::cout << sub.get<BatchedScalar>("s2") << std::endl;

  // Derivative wrt self
  StateDerivative AA(state, state);

  std::cout << state.sizes() << std::endl;
  std::cout << AA.sizes() << std::endl;

  std::cout << AA.get<BatchedScalar>("s1","s3") << std::endl;

  std::cout << AA.get<BatchedSymSymR4>("sym1", "sym2") << std::endl;
}
