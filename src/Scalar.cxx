#include "Scalar.h"

Scalar::Scalar() :
    ScalarBase(torch::zeros({1}))
{

}

Scalar::Scalar(const torch::Tensor & tensor) :
    ScalarBase(tensor)
{

}

Scalar::Scalar(const double & other) :
    ScalarBase(torch::empty({1}))
{
  index_put_({0}, other);
}

double Scalar::value() const
{
  return item<double>();
}
