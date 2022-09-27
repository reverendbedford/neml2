#include "SymSymR4.h"

using namespace torch::indexing;

SymSymR4::SymSymR4()
  : SymSymR4Base(torch::zeros({6, 6}, TorchDefaults))
{
}

SymSymR4::SymSymR4(const torch::Tensor & tensor)
  : SymSymR4Base(tensor)
{
}

SymR2
SymSymR4::dot(const SymR2 & b)
{
  return torch::matmul(*this, b);
}

BatchedSymR2
SymSymR4::dot(const BatchedSymR2 & b)
{
  // Lol, but yes, read the docs for matmul
  return torch::matmul(b, *this);
}
