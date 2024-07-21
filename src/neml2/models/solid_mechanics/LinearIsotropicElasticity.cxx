// Copyright 2023, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "neml2/models/solid_mechanics/LinearIsotropicElasticity.h"
#include "neml2/tensors/SSR4.h"
#include "neml2/misc/math.h"

namespace neml2
{
register_NEML2_object(LinearIsotropicElasticity);

OptionSet
LinearIsotropicElasticity::expected_options()
{
  OptionSet options = Elasticity::expected_options();
  options.doc() += " for linear isotropic material. \\f$ \\boldsymbol{\\sigma} = K \\tr "
                   "\\boldsymbol{\\varepsilon}_e + 2 G \\text{dev} \\boldsymbol{\\varepsilon}_e "
                   "\\f$, where \\f$ K \\f$ and \\f$ G \\f$ are bulk and shear moduli, "
                   "respectively. For convenience, this object only requests Young's modulus and "
                   "Poisson's ratio, and handles the Lame parameter conversion behind the scenes.";

  options.set_parameter<CrossRef<Scalar>>("youngs_modulus");
  options.set("youngs_modulus").doc() = "Young's modulus";

  options.set_parameter<CrossRef<Scalar>>("poisson_ratio");
  options.set("poisson_ratio").doc() = "Poisson's ratio";

  return options;
}

LinearIsotropicElasticity::LinearIsotropicElasticity(const OptionSet & options)
  : Elasticity(options),
    _E(declare_parameter<Scalar>("E", "youngs_modulus", /*allow_nonlinear=*/true)),
    _nu(declare_parameter<Scalar>("nu", "poisson_ratio", /*allow_nonlinear=*/true))
{
}

void
LinearIsotropicElasticity::set_value(bool out, bool dout_din, bool d2out_din2)
{
  // We need to work with the bulk modulus K and the shear modulus G so that the expression for
  // stiffness and compliance can be unified:
  const auto K = _E / 3 / (1 - 2 * _nu);
  const auto G = _E / 2 / (1 + _nu);
  const auto vf = _compliance ? 1 / (3 * K) : 3 * K;
  const auto df = _compliance ? 1 / (2 * G) : 2 * G;

  if (out)
    _to = vf * SR2(_from).vol() + df * SR2(_from).dev();

  const auto E = nl_param("E");
  const auto nu = nl_param("nu");

  if (dout_din)
  {
    const auto I = SSR4::identity_vol(options());
    const auto J = SSR4::identity_dev(options());
    _to.d(_from) = vf * I + df * J;

    if (E)
    {
      const auto dvf_dE = _compliance ? -(1 - 2 * _nu) / _E / _E : 1 / (1 - 2 * _nu);
      const auto ddf_dE = _compliance ? -(1 + _nu) / _E / _E : 1 / (1 + _nu);
      _to.d(*E) = dvf_dE * SR2(_from).vol() + ddf_dE * SR2(_from).dev();
    }

    if (nu)
    {
      const auto dvf_dnu = _compliance ? -2 / _E : 2 * _E / (1 - 2 * _nu) / (1 - 2 * _nu);
      const auto ddf_dnu = _compliance ? 1 / _E : -_E / (1 + _nu) / (1 + _nu);
      _to.d(*nu) = dvf_dnu * SR2(_from).vol() + ddf_dnu * SR2(_from).dev();
    }
  }

  if (d2out_din2)
  {
    const auto I = SSR4::identity_vol(options());
    const auto J = SSR4::identity_dev(options());

    if (E)
    {
      const auto dvf_dE = _compliance ? -(1 - 2 * _nu) / _E / _E : 1 / (1 - 2 * _nu);
      const auto ddf_dE = _compliance ? -(1 + _nu) / _E / _E : 1 / (1 + _nu);
      const auto deriv = dvf_dE * I + ddf_dE * J;
      _to.d(_from, *E) = deriv;
      _to.d(*E, _from) = deriv;
    }

    if (nu)
    {
      const auto dvf_dnu = _compliance ? -2 / _E : 2 * _E / (1 - 2 * _nu) / (1 - 2 * _nu);
      const auto ddf_dnu = _compliance ? 1 / _E : -_E / (1 + _nu) / (1 + _nu);
      const auto deriv = dvf_dnu * I + ddf_dnu * J;
      _to.d(_from, *nu) = deriv;
      _to.d(*nu, _from) = deriv;
    }

    if (E && _compliance)
    {
      const auto d2vf_dE2 = 2 * (1 - 2 * _nu) / _E / _E / _E;
      const auto d2df_dE2 = 2 * (1 + _nu) / _E / _E / _E;
      _to.d(*E, *E) = d2vf_dE2 * SR2(_from).vol() + d2df_dE2 * SR2(_from).dev();
    }

    if (nu && !_compliance)
    {
      const auto d2vf_dnu2 = 8 * _E / math::pow(1 - 2 * _nu, 3);
      const auto d2df_dnu2 = 2 * _E / math::pow(1 + _nu, 3);
      _to.d(*nu, *nu) = d2vf_dnu2 * SR2(_from).vol() + d2df_dnu2 * SR2(_from).dev();
    }

    if (E && nu)
    {
      const auto d2vf_dEdnu = _compliance ? 2 / _E / _E : 2 / (1 - 2 * _nu) / (1 - 2 * _nu);
      const auto d2df_dEdnu = _compliance ? -1 / _E / _E : -1 / (1 + _nu) / (1 + _nu);
      const auto deriv = d2vf_dEdnu * SR2(_from).vol() + d2df_dEdnu * SR2(_from).dev();
      _to.d(*E, *nu) = deriv;
      _to.d(*nu, *E) = deriv;
    }
  }
}
} // namespace neml2
