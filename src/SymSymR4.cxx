#include "SymSymR4.h"

SymSymR4
SymSymR4::init(SymSymR4::FillMethod method, const std::vector<Scalar> & vals)
{
  switch (method)
  {
    case SymSymR4::FillMethod::identity_sym:
      return SymSymR4::init_identity_sym();
    case SymSymR4::FillMethod::identity_vol:
      return SymSymR4::init_identity() / 3;
    case SymSymR4::FillMethod::identity_dev:
      return SymSymR4::init_identity_sym() - SymSymR4::init_identity() / 3;
    case SymSymR4::FillMethod::isotropic_E_nu:
      return SymSymR4::init_isotropic_E_nu(vals[0], vals[1]);
    default:
      std::runtime_error("Unsupported fill method");
      return SymSymR4();
  }
}

SymSymR4
SymSymR4::init_identity()
{
  return SymSymR4(torch::tensor({{1, 1, 1, 0, 0, 0},
                                 {1, 1, 1, 0, 0, 0},
                                 {1, 1, 1, 0, 0, 0},
                                 {0, 0, 0, 0, 0, 0},
                                 {0, 0, 0, 0, 0, 0},
                                 {0, 0, 0, 0, 0, 0}},
                                TorchDefaults),
                  1);
}

SymSymR4
SymSymR4::init_identity_sym()
{
  return SymSymR4(torch::eye(6), 1);
}

SymSymR4
SymSymR4::init_isotropic_E_nu(const Scalar & E, const Scalar & nu)
{
  SymSymR4 C;
  C = C.expand_batch(E.batch_sizes());

  Scalar pf = E / ((1 + nu) * (1 - 2 * nu));
  Scalar C1 = (1 - nu) * pf;
  Scalar C2 = nu * pf;
  Scalar C4 = (1 - 2 * nu) * pf;

  for (TorchSize i = 0; i < 3; i++)
  {
    for (TorchSize j = 0; j < 3; j++)
    {
      if (i == j)
        C.base_index_put({i, j}, C1.squeeze(-1));
      else
        C.base_index_put({i, j}, C2.squeeze(-1));
    }
  }

  for (TorchSize i = 3; i < 6; i++)
    C.base_index_put({i, i}, C4.squeeze(-1));

  return C;
}

SymR2
SymSymR4::operator*(const SymR2 & b)
{
  return torch::matmul(*this, b.unsqueeze(2)).squeeze(2);
}

SymSymR4
SymSymR4::operator*(const SymSymR4 & b)
{
  return torch::matmul(*this, b);
}
