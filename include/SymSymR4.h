#include <torch/torch.h>

class SymSymR4 : public torch::Tensor {
 public:
  SymSymR4();
  SymSymR4(const torch::Tensor & tensor);
};
