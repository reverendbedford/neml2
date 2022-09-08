#include <torch/torch.h>
#include <iostream>

#include "LabeledTensor.h"
#include "BatchTensor.h"
#include "FixedDimTensor.h"

using namespace torch::indexing;

int main()
{
  LabeledTensor<0> ten(torch::zeros({5,6}));
  std::cout << ten << std::endl;

  ten.add_label("blah", {Slice(), 1});
  
  auto sect = ten.get("blah");

  sect.fill_(1.0);
  std::cout << sect << std::endl;
  std::cout << ten << std::endl;

  BatchTensor<0> b0(torch::zeros({10,5}));
  std::cout << b0.nbatch() << std::endl;
  std::cout << b0.batch_sizes() << std::endl;
  std::cout << b0.base_sizes() << std::endl;

  BatchTensor<1> b1(torch::zeros({10,5}));
  std::cout << b1.nbatch() << std::endl;
  std::cout << b1.batch_sizes() << std::endl;
  std::cout << b1.base_sizes() << std::endl;
  
  FixedDimTensor<2, 3, 4, 5> test(torch::zeros({10,9,3,4,5}));
  std::cout << test.nbatch() << std::endl;
  std::cout << test.batch_sizes() << std::endl;
  std::cout << test.base_sizes() << std::endl;

  FixedDimTensor<2, 3, 4, 5> test2({10,9});
  std::cout << test2.nbatch() << std::endl;
  std::cout << test2.batch_sizes() << std::endl;
  std::cout << test2.base_sizes() << std::endl;
}
