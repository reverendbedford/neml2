#include <torch/torch.h>

class Scalar : public torch::Tensor {
 public:
  Scalar();
  Scalar(const torch::Tensor & tensor);
};
