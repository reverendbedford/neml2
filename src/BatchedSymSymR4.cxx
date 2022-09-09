#include "BatchedSymSymR4.h"

BatchedSymSymR4::BatchedSymSymR4(TorchSize nbatch) :
    BatchedSymSymR4Base(nbatch)
{

}

BatchedSymSymR4::BatchedSymSymR4(const torch::Tensor & tensor) :
    BatchedSymSymR4Base(tensor)
{
  
}
