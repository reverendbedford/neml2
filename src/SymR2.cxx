#include "SymR2.h"

SymR2::SymR2() :
    torch::Tensor(torch::zeros({6}))
{

}

SymR2::SymR2(const torch::Tensor & tensor) :
    torch::Tensor(tensor)
{
  assert(tensor.sizes() == (torch::IntArrayRef{6}));
}
