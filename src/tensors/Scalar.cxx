#include "tensors/Scalar.h"

namespace neml2
{
Scalar::Scalar(double init, TorchSize batch_size)
  : FixedDimTensor<1, 1>(torch::tensor(init, TorchDefaults), batch_size)
{
}

Scalar
Scalar::operator-() const
{
  return -torch::Tensor(*this);
}

Scalar
operator+(const Scalar & a, const Scalar & b)
{
  return torch::operator+(a, b);
}

BatchTensor<1>
operator+(const BatchTensor<1> & a, const Scalar & b)
{
  torch::Tensor tmp = b;
  for (TorchSize i = 1; i < a.base_dim(); i++)
    tmp = tmp.unsqueeze(-1);
  return torch::operator+(a, tmp);
}

BatchTensor<1>
operator+(const Scalar & a, const BatchTensor<1> & b)
{
  return b + a;
}

Scalar
operator-(const Scalar & a, const Scalar & b)
{
  return torch::operator-(a, b);
}

BatchTensor<1>
operator-(const BatchTensor<1> & a, const Scalar & b)
{
  torch::Tensor tmp = b;
  for (TorchSize i = 1; i < a.base_dim(); i++)
    tmp = tmp.unsqueeze(-1);
  return torch::operator-(a, tmp);
}

BatchTensor<1>
operator-(const Scalar & a, const BatchTensor<1> & b)
{
  return -b + a;
}

Scalar
operator*(const Scalar & a, const Scalar & b)
{
  return torch::operator*(a, b);
}

BatchTensor<1>
operator*(const BatchTensor<1> & a, const Scalar & b)
{
  torch::Tensor tmp = b;
  for (TorchSize i = 1; i < a.base_dim(); i++)
    tmp = tmp.unsqueeze(-1);
  return torch::operator*(a, tmp);
}

BatchTensor<1>
operator*(const Scalar & a, const BatchTensor<1> & b)
{
  return b * a;
}

Scalar
operator/(const Scalar & a, const Scalar & b)
{
  return torch::operator/(a, b);
}

BatchTensor<1>
operator/(const BatchTensor<1> & a, const Scalar & b)
{
  torch::Tensor tmp = b;
  for (TorchSize i = 1; i < a.base_dim(); i++)
    tmp = tmp.unsqueeze(-1);
  return torch::operator/(a, tmp);
}

BatchTensor<1>
operator/(const Scalar & a, const BatchTensor<1> & b)
{
  torch::Tensor tmp = a;
  for (TorchSize i = 1; i < a.base_dim(); i++)
    tmp = tmp.unsqueeze(-1);
  return torch::operator/(tmp, b);
}

Scalar
macaulay(const Scalar & a, const Scalar & a0)
{
  return a * Scalar(torch::heaviside(a, a0));
}

/// Derivative of the Macaulay bracket
Scalar
dmacaulay(const Scalar & a, const Scalar & a0)
{
  return torch::heaviside(a, a0);
}
} // namespace neml2
