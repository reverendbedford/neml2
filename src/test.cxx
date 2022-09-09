#include <torch/torch.h>
#include <iostream>

#include "State.h"
#include "BatchedScalar.h"
#include "BatchedSymR2.h"

#include "LabeledTensor.h"
#include "BatchTensor.h"
#include "FixedDimTensor.h"
#include "BatchedScalar.h"

using namespace torch::indexing;

int main()
{
  TorchSize nbatch = 10;
  // A scalar, a SymR2, and a scalar
  State state(torch::zeros({nbatch, 8}));
  state.add_label("one", {0});
  state.add_label("two", {Slice(1,7)});
  state.add_label("three", {7});
  
  std::cout << state.sizes() << std::endl;
  std::cout << state.view("one").sizes() << std::endl;
  std::cout << state.view("two").sizes() << std::endl;
  std::cout << state.view("three").sizes() << std::endl;
}
