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

#include "neml2/models/solid_mechanics/GTNYieldFunction.h"

namespace neml2
{
register_NEML2_object(GTNYieldFunction);

OptionSet
GTNYieldFunction::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<CrossRef<Scalar>>("yield_stress");
  options.set<CrossRef<Scalar>>("q1");
  options.set<CrossRef<Scalar>>("q2");
  options.set<CrossRef<Scalar>>("q3");
  options.set<VariableName>("flow_invariant") = {{"state", "internal", "se"}};
  options.set<VariableName>("poro_invariant") = {{"state", "internal", "sp"}};
  options.set<VariableName>("isotropic_hardening");
  options.set<VariableName>("void_fraction") = {{"state", "internal", "f"}};
  options.set<VariableName>("yield_function") = {{"state", "internal", "fp"}};
  return options;
}

GTNYieldFunction::GTNYieldFunction(const OptionSet & options)
  : Model(options),
    _f(declare_output_variable<Scalar>(options.get<VariableName>("yield_function"))),
    _se(declare_input_variable<Scalar>(options.get<VariableName>("flow_invariant"))),
    _sp(declare_input_variable<Scalar>(options.get<VariableName>("poro_invariant"))),
    _phi(declare_input_variable<Scalar>(options.get<VariableName>("void_fraction"))),
    _h(options.get<VariableName>("isotropic_hardening").empty()
           ? nullptr
           : &declare_input_variable<Scalar>(options.get<VariableName>("isotropic_hardening"))),
    _s0(declare_parameter<Scalar>("sy", "yield_stress")),
    _q1(declare_parameter<Scalar>("q1", "q1")),
    _q2(declare_parameter<Scalar>("q2", "q2")),
    _q3(declare_parameter<Scalar>("q3", "q3"))
{
}

void
GTNYieldFunction::set_value(bool out, bool dout_din, bool d2out_din2)
{
  // Flow stress (depending on whether isotropic hardening is provided)
  const auto sf = _h ? _s0 + (*_h) : _s0;

  if (out)
    _f = math::pow(_se / sf, 2.0) + 2 * _q1 * _phi * math::cosh(_q2 / 2.0 * _sp / sf) -
         (1.0 + _q3 * math::pow(Scalar(_phi), 2.0));

  if (dout_din)
  {
    _f.d(_se) = 2.0 * _se / math::pow(sf, 2.0);
    _f.d(_sp) = _q1 * _phi * _q2 / sf * math::sinh(_q2 / 2.0 * _sp / sf);
    _f.d(_phi) = 2.0 * _q1 * math::cosh(_q2 / 2.0 * _sp / sf) - 2.0 * _q3 * _phi;
    if (_h)
      _f.d(*_h) = -2 * math::pow(Scalar(_se), 2.0) / math::pow(sf, 3.0) -
                  _q1 * _phi * _q2 * _sp / math::pow(sf, 2.0) * math::sinh(_q2 / 2.0 * _sp / sf);

    // Handle the case of nonlinear parameters
    if (const auto sy = nl_param("sy"))
      _f.d(*sy) = -2 * math::pow(Scalar(_se), 2.0) / math::pow(sf, 3.0) -
                  _q1 * _phi * _q2 * _sp / math::pow(sf, 2.0) * math::sinh(_q2 / 2.0 * _sp / sf);

    if (const auto q1 = nl_param("q1"))
      _f.d(*q1) = 2.0 * _phi * math::cosh(_q2 / 2.0 * _sp / sf);

    if (const auto q2 = nl_param("q2"))
      _f.d(*q2) = _q1 * _phi * _sp / sf * math::sinh(_q2 / 2.0 * _sp / sf);

    if (const auto q3 = nl_param("q3"))
      _f.d(*q3) = -math::pow(Scalar(_phi), 2.0);
  }

  if (d2out_din2)
  {
    const auto sy = nl_param("sy");
    const auto q1 = nl_param("q1");
    const auto q2 = nl_param("q2");
    const auto q3 = nl_param("q3");

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // The GTN yield function can be expressed as
    //
    //      f(se, sp, phi, h; sy, q1, q2, q3)
    //
    //  - Arguments before the semicolon are variables
    //  - Arguments after the semicolon are (nonlinear) parameters
    //  - Derivatives w.r.t. the first three arguments se, sp, and phi are mandatory
    //  - Derivatives w.r.t. the rest of the arguments are optional
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // The second derivative is nothing but a big matrix. We will fill out the matrix row by row, in
    // the order of the arguments listed above.
    //
    // Rows will separated by big fences like this.
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // se: Flow invariant
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    _f.d(_se, _se) = 2.0 / math::pow(sf, 2.0);

    if (_h)
      _f.d(_se, *_h) = -4.0 * _se / math::pow(sf, 3.0);

    if (sy)
      _f.d(_se, *sy) = -4.0 * _se / math::pow(sf, 3.0);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // sp: Poro invariant
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    _f.d(_sp, _sp) = _phi * _q1 * math::pow(_q2, 2.0) / (2.0 * math::pow(sf, 2.0)) *
                     math::cosh(_q2 / 2.0 * _sp / sf);

    _f.d(_sp, _phi) = _q1 * _q2 * math::sinh(_q2 / 2.0 * _sp / sf) / sf;

    if (_h)
      _f.d(_sp, *_h) = -_phi * _q1 * _q2 *
                       (_q2 * _sp * math::cosh(_q2 / 2.0 * _sp / sf) +
                        2 * sf * math::sinh(_q2 / 2.0 * _sp / sf)) /
                       (2 * math::pow(sf, 3.0));
    if (sy)
      _f.d(_sp, *sy) = -_phi * _q1 * _q2 *
                       (_q2 * _sp * math::cosh(_q2 / 2.0 * _sp / sf) +
                        2 * sf * math::sinh(_q2 / 2.0 * _sp / sf)) /
                       (2 * math::pow(sf, 3.0));

    if (q1)
      _f.d(_sp, *q1) = _phi * _q2 * math::sinh(_q2 / 2.0 * _sp / sf) / sf;

    if (q2)
      _f.d(_sp, *q2) = _phi * _q1 *
                       (_q2 * _sp * math::cosh(_q2 / 2.0 * _sp / sf) +
                        2.0 * sf * math::sinh(_q2 / 2.0 * _sp / sf)) /
                       (2.0 * math::pow(sf, 2.0));

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // phi: Void fraction
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    _f.d(_phi, _sp) = _q1 * _q2 * math::sinh(_q2 / 2.0 * _sp / sf) / sf;

    _f.d(_phi, _phi) = -2.0 * _q3;

    if (_h)
      _f.d(_phi, *_h) = -_q1 * _q2 * _sp * math::sinh(_q2 / 2.0 * _sp / sf) / math::pow(sf, 2.0);

    if (sy)
      _f.d(_phi, *sy) = -_q1 * _q2 * _sp * math::sinh(_q2 / 2.0 * _sp / sf) / math::pow(sf, 2.0);

    if (q1)
      _f.d(_phi, *q1) = 2 * math::cosh(_q2 / 2.0 * _sp / sf);

    if (q2)
      _f.d(_phi, *q2) = _q1 * _sp * math::sinh(_q2 / 2.0 * _sp / sf) / sf;

    if (q3)
      _f.d(_phi, *q3) = -2.0 * _phi;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // h: (Optional) isotropic hardening
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (_h)
    {
      _f.d(*_h, _se) = -4.0 * _se / math::pow(sf, 3.0);

      _f.d(*_h, _sp) = -_phi * _q1 * _q2 *
                       (_q2 * _sp * math::cosh(_q2 / 2.0 * _sp / sf) +
                        2.0 * sf * math::sinh(_q2 / 2.0 * _sp / sf)) /
                       (2.0 * math::pow(sf, 3.0));

      _f.d(*_h, _phi) = -_q1 * _q2 * _sp * math::sinh(_q2 / 2.0 * _sp / sf) / math::pow(sf, 2.0);

      _f.d(*_h, *_h) =
          (12 * math::pow(Scalar(_se), 2.0) + _phi * _q1 * _q2 * _sp *
                                                  (_q2 * _sp * math::cosh(_q2 / 2.0 * _sp / sf) +
                                                   4.0 * sf * math::sinh(_q2 / 2.0 * _sp / sf))) /
          (2 * math::pow(sf, 4.0));

      if (sy)
        _f.d(*_h, *sy) =
            (12 * math::pow(Scalar(_se), 2.0) + _phi * _q1 * _q2 * _sp *
                                                    (_q2 * _sp * math::cosh(_q2 / 2.0 * _sp / sf) +
                                                     4.0 * sf * math::sinh(_q2 / 2.0 * _sp / sf))) /
            (2 * math::pow(sf, 4.0));

      if (q1)
        _f.d(*_h, *q1) = -_phi * _q2 * _sp * math::sinh(_q2 / 2.0 * _sp / sf) / math::pow(sf, 2.0);

      if (q2)
        _f.d(*_h, *q2) = -_phi * _q1 * _sp *
                         (_q2 * _sp * math::cosh(_q2 / 2.0 * _sp / sf) +
                          2.0 * sf * math::sinh(_q2 / 2.0 * _sp / sf)) /
                         (2 * math::pow(sf, 3.0));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // sy: (Optionally nonlinear) yield stress
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (sy)
    {
      _f.d(*sy, _se) = -4.0 * _se / math::pow(sf, 3.0);

      _f.d(*sy, _phi) = -_q1 * _q2 * _sp * math::sinh(_q2 / 2.0 * _sp / sf) / math::pow(sf, 2.0);

      if (_h)
        _f.d(*sy, *_h) =
            (12 * math::pow(Scalar(_se), 2.0) + _phi * _q1 * _q2 * _sp *
                                                    (_q2 * _sp * math::cosh(_q2 / 2.0 * _sp / sf) +
                                                     4.0 * sf * math::sinh(_q2 / 2.0 * _sp / sf))) /
            (2 * math::pow(sf, 4.0));

      _f.d(*sy, *sy) =
          (12 * math::pow(Scalar(_se), 2.0) + _phi * _q1 * _q2 * _sp *
                                                  (_q2 * _sp * math::cosh(_q2 / 2.0 * _sp / sf) +
                                                   4.0 * sf * math::sinh(_q2 / 2.0 * _sp / sf))) /
          (2 * math::pow(sf, 4.0));

      if (q1)
        _f.d(*sy, *q1) = -_phi * _q2 * _sp * math::sinh(_q2 / 2.0 * _sp / sf) / math::pow(sf, 2.0);

      if (q2)
        _f.d(*sy, *q2) = -_phi * _q1 * _sp *
                         (_q2 * _sp * math::cosh(_q2 / 2.0 * _sp / sf) +
                          2.0 * sf * math::sinh(_q2 / 2.0 * _sp / sf)) /
                         (2 * math::pow(sf, 3.0));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // q1: (Optionally nonlinear) GTN parameter q1
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (q1)
    {
      _f.d(*q1, _sp) = _phi * _q2 * math::sinh(_q2 / 2.0 * _sp / sf) / sf;

      _f.d(*q1, _phi) = 2.0 * math::cosh(_q2 / 2.0 * _sp / sf);

      if (_h)
        _f.d(*q1, *_h) = -_phi * _q2 * _sp * math::sinh(_q2 / 2.0 * _sp / sf) / math::pow(sf, 2.0);

      if (sy)
        _f.d(*q1, *sy) = -_phi * _q2 * _sp * math::sinh(_q2 / 2.0 * _sp / sf) / math::pow(sf, 2.0);

      if (q2)
        _f.d(*q1, *q2) = _phi * _sp * math::sinh(_q2 / 2.0 * _sp / sf) / sf;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // q2: (Optionally nonlinear) GTN parameter q2
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (q2)
    {
      _f.d(*q2, _sp) = _phi * _q1 *
                       (_q2 * _sp * math::cosh(_q2 / 2.0 * _sp / sf) +
                        2 * sf * math::sinh(_q2 / 2.0 * _sp / sf)) /
                       (2 * math::pow(sf, 2.0));

      _f.d(*q2, _phi) = _q1 * _sp * math::sinh(_q2 / 2.0 * _sp / sf) / sf;

      if (_h)
        _f.d(*q2, *_h) = -_phi * _q1 * _sp *
                         (_q2 * _sp * math::cosh(_q2 / 2.0 * _sp / sf) +
                          2 * sf * math::sinh(_q2 / 2.0 * _sp / sf)) /
                         (2 * math::pow(sf, 3.0));

      if (sy)
        _f.d(*q2, *sy) = -_phi * _q1 * _sp *
                         (_q2 * _sp * math::cosh(_q2 / 2.0 * _sp / sf) +
                          2 * sf * math::sinh(_q2 / 2.0 * _sp / sf)) /
                         (2 * math::pow(sf, 3.0));

      if (q1)
        _f.d(*q2, *q1) = _phi * _sp * math::sinh(_q2 / 2.0 * _sp / sf) / sf;

      _f.d(*q2, *q2) = _phi * _q1 * math::pow(Scalar(_sp), 2.0) * math::cosh(_q2 / 2.0 * _sp / sf) /
                       (2 * math::pow(sf, 2.0));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // f(se, sp, phi, h; sy, q1, q2, q3)
    //
    // q3: (Optionally nonlinear) GTN parameter q3
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////
    if (q3)
    {
      _f.d(*q3, _phi) = -2.0 * _phi;
    }
  }
}
} // namespace neml2
