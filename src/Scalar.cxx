#include "Scalar.h"

Scalar::Scalar() :
    ScalarBase()
{

}

Scalar::Scalar(const torch::Tensor & tensor) :
    ScalarBase(tensor)
{

}
