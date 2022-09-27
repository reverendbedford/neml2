#include "ElasticityTensors.h"

SymSymR4
fill_isotropic(const Scalar & E, const Scalar & nu)
{
  SymSymR4 C;

  Scalar pf = E / ((1 + nu) * (1 - 2 * nu));
  Scalar C1 = (1 - nu) * pf;
  Scalar C2 = nu * pf;
  Scalar C4 = (1 - 2 * nu) * pf;

  for (TorchSize i = 0; i < 3; i++)
  {
    for (TorchSize j = 0; j < 3; j++)
    {
      if (i == j)
        C.index_put_({i, j}, C1);
      else
        C.index_put_({i, j}, C2);
    }
  }

  for (TorchSize i = 3; i < 6; i++)
    C.index_put_({i, i}, C4);

  return C;
}
