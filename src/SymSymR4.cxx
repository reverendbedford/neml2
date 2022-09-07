#include "SymSymR4.h"

SymSymR4::SymSymR4() :
    torch::Tensor(torch::zeros({1}))
{

}

SymSymR4::SymSymR4(const torch::Tensor & tensor) :
    torch::Tensor(tensor)
{
  assert(tensor.sizes() == (torch::IntArrayRef{6,6}));
}
