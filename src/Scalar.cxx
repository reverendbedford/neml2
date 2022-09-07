#include "Scalar.h"

Scalar::Scalar() :
    torch::Tensor(torch::zeros({1}))
{

}

Scalar::Scalar(const torch::Tensor & tensor) :
    torch::Tensor(tensor)
{
  assert(tensor.sizes() == (torch::IntArrayRef{1}));
}
