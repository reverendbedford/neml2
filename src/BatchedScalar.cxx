#include "BatchedScalar.h"

BatchedScalar::BatchedScalar(TorchSize nbatch) :
    BatchedScalarBase(nbatch)
{

}

BatchedScalar::BatchedScalar(const torch::Tensor & tensor) :
    BatchedScalarBase(tensor)
{

}
