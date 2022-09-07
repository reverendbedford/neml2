#include <torch/torch.h>
#include <iostream>

#include "LabeledTensor.h"
#include "BatchTensor.h"

using namespace torch::indexing;

int main()
{
  LabeledTensor ten(torch::zeros({5,6}));
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

  BatchTensor<3> b2(torch::zeros({10,5}));
}
