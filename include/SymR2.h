#include <torch/torch.h>

class SymR2 : public torch::Tensor {
 public:
  SymR2();
  SymR2(const torch::Tensor & tensor);
};
