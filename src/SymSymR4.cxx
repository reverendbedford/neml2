#include "SymSymR4.h"

SymSymR4::SymSymR4() :
    SymSymR4Base(torch::zeros({6,6}))
{

}

SymSymR4::SymSymR4(const torch::Tensor & tensor) :
    SymSymR4Base(tensor)
{

}
