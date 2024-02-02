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
  options.set<LabeledAxisAccessor>("flow_invariant") = {{"state", "internal", "se"}};
  options.set<LabeledAxisAccessor>("poro_invariant") = {{"state", "internal", "sp"}};
  options.set<LabeledAxisAccessor>("isotropic_hardening");
  options.set<LabeledAxisAccessor>("void_fraction") = {{"state", "internal", "f"}};
  options.set<LabeledAxisAccessor>("yield_function") = {{"state", "internal", "fp"}};
  return options;
}

GTNYieldFunction::GTNYieldFunction(const OptionSet & options)
  : Model(options),
    flow_invariant(
        declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("flow_invariant"))),
    poro_invariant(
        declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("poro_invariant"))),
    isotropic_hardening(options.get<LabeledAxisAccessor>("isotropic_hardening")),
    void_fraction(
        declare_input_variable<Scalar>(options.get<LabeledAxisAccessor>("void_fraction"))),
    yield_function(
        declare_output_variable<Scalar>(options.get<LabeledAxisAccessor>("yield_function"))),
    _s0(declare_parameter<Scalar>("sy", "yield_stress")),
    _q1(declare_parameter<Scalar>("q1", "q1")),
    _q2(declare_parameter<Scalar>("q2", "q2")),
    _q3(declare_parameter<Scalar>("q3", "q3"))
{
  if (!isotropic_hardening.empty())
    declare_input_variable<Scalar>(isotropic_hardening);
  setup();
}

void
GTNYieldFunction::set_value(const LabeledVector & in,
                            LabeledVector * out,
                            LabeledMatrix * dout_din,
                            LabeledTensor3D * d2out_din2) const
{
  auto se = in.get<Scalar>(flow_invariant);
  auto sp = in.get<Scalar>(poro_invariant);
  auto f = in.get<Scalar>(void_fraction);
  Scalar sf;
  if (!isotropic_hardening.empty())
    sf = _s0 + in.get<Scalar>(isotropic_hardening);
  else
    sf = _s0;

  if (out)
  {
    out->set(math::pow(se / sf, 2.0) + 2 * _q1 * f * math::cosh(_q2 / 2.0 * sp / sf) -
                 (1.0 + _q3 * math::pow(f, 2.0)),
             yield_function);
  }

  if (dout_din)
  {
    auto I = Scalar::identity_map(in.options());
    dout_din->set(2.0 * se / math::pow(sf, 2.0) * I, yield_function, flow_invariant);
    dout_din->set(
        _q1 * f * _q2 / sf * math::sinh(_q2 / 2.0 * sp / sf) * I, yield_function, poro_invariant);
    dout_din->set((2.0 * _q1 * math::cosh(_q2 / 2.0 * sp / sf) - 2.0 * _q3 * f) * I,
                  yield_function,
                  void_fraction);
    if (!isotropic_hardening.empty())
      dout_din->set((-2 * math::pow(se, 2.0) / math::pow(sf, 3.0) -
                     _q1 * f * _q2 * sp / math::pow(sf, 2.0) * math::sinh(_q2 / 2.0 * sp / sf)) *
                        I,
                    yield_function,
                    isotropic_hardening);
    if (has_nonlinear_parameter("sy"))
      dout_din->set((-2 * math::pow(se, 2.0) / math::pow(sf, 3.0) -
                     _q1 * f * _q2 * sp / math::pow(sf, 2.0) * math::sinh(_q2 / 2.0 * sp / sf)) *
                        I,
                    yield_function,
                    nl_param("sy"));
    if (has_nonlinear_parameter("q1"))
      dout_din->set(2.0 * f * math::cosh(_q2 / 2.0 * sp / sf) * I, yield_function, nl_param("q1"));
    if (has_nonlinear_parameter("q2"))
      dout_din->set(
          _q1 * f * sp / sf * math::sinh(_q2 / 2.0 * sp / sf) * I, yield_function, nl_param("q2"));
    if (has_nonlinear_parameter("q3"))
      dout_din->set(-math::pow(f, 2.0) * I, yield_function, nl_param("q3"));
  }

  if (d2out_din2)
  {
    auto I = Scalar::identity_map(in.options());
    // So much pain...  Go one at a time
    // flow invariant
    // flow invariant
    d2out_din2->set(2.0 / math::pow(sf, 2.0) * I, yield_function, flow_invariant, flow_invariant);
    // isotropic hardening
    if (!isotropic_hardening.empty())
      d2out_din2->set(
          -4.0 * se / math::pow(sf, 3.0) * I, yield_function, flow_invariant, isotropic_hardening);
    // yield stress
    if (has_nonlinear_parameter("sy"))
      d2out_din2->set(
          -4.0 * se / math::pow(sf, 3.0) * I, yield_function, flow_invariant, nl_param("sy"));

    // poro invariant
    // poro invariant
    d2out_din2->set(f * _q1 * math::pow(_q2, 2.0) / (2.0 * math::pow(sf, 2.0)) *
                        math::cosh(_q2 / 2.0 * sp / sf) * I,
                    yield_function,
                    poro_invariant,
                    poro_invariant);
    // void fraction
    d2out_din2->set(_q1 * _q2 * math::sinh(_q2 / 2.0 * sp / sf) / sf * I,
                    yield_function,
                    poro_invariant,
                    void_fraction);
    // isotropic hardening
    if (!isotropic_hardening.empty())
      d2out_din2->set(-f * _q1 * _q2 *
                          (_q2 * sp * math::cosh(_q2 / 2.0 * sp / sf) +
                           2 * sf * math::sinh(_q2 / 2.0 * sp / sf)) /
                          (2 * math::pow(sf, 3.0)) * I,
                      yield_function,
                      poro_invariant,
                      isotropic_hardening);
    // yield stress
    if (has_nonlinear_parameter("sy"))
      d2out_din2->set(-f * _q1 * _q2 *
                          (_q2 * sp * math::cosh(_q2 / 2.0 * sp / sf) +
                           2 * sf * math::sinh(_q2 / 2.0 * sp / sf)) /
                          (2 * math::pow(sf, 3.0)) * I,
                      yield_function,
                      poro_invariant,
                      nl_param("sy"));
    // q1
    if (has_nonlinear_parameter("q1"))
      d2out_din2->set(f * _q2 * math::sinh(_q2 / 2.0 * sp / sf) / sf * I,
                      yield_function,
                      poro_invariant,
                      nl_param("q1"));

    // q2
    if (has_nonlinear_parameter("q2"))
      d2out_din2->set(f * _q1 *
                          (_q2 * sp * math::cosh(_q2 / 2.0 * sp / sf) +
                           2.0 * sf * math::sinh(_q2 / 2.0 * sp / sf)) /
                          (2.0 * math::pow(sf, 2.0)) * I,
                      yield_function,
                      poro_invariant,
                      nl_param("q2"));

    // void fraction
    // poro invariant
    d2out_din2->set(_q1 * _q2 * math::sinh(_q2 / 2.0 * sp / sf) / sf * I,
                    yield_function,
                    void_fraction,
                    poro_invariant);
    // void fraction
    d2out_din2->set(-2.0 * _q3 * I, yield_function, void_fraction, void_fraction);
    // isotropic hardening
    if (!isotropic_hardening.empty())
      d2out_din2->set(-_q1 * _q2 * sp * math::sinh(_q2 / 2.0 * sp / sf) / math::pow(sf, 2.0) * I,
                      yield_function,
                      void_fraction,
                      isotropic_hardening);
    // sy
    if (has_nonlinear_parameter("sy"))
      d2out_din2->set(-_q1 * _q2 * sp * math::sinh(_q2 / 2.0 * sp / sf) / math::pow(sf, 2.0) * I,
                      yield_function,
                      void_fraction,
                      nl_param("sy"));
    // q1
    if (has_nonlinear_parameter("q1"))
      d2out_din2->set(
          2 * math::cosh(_q2 / 2.0 * sp / sf) * I, yield_function, void_fraction, nl_param("q1"));
    // q2
    if (has_nonlinear_parameter("q2"))
      d2out_din2->set(_q1 * sp * math::sinh(_q2 / 2.0 * sp / sf) / sf * I,
                      yield_function,
                      void_fraction,
                      nl_param("q2"));
    // q3
    if (has_nonlinear_parameter("q3"))
      d2out_din2->set(-2.0 * f * I, yield_function, void_fraction, nl_param("q3"));

    // isotropic hardening
    if (!isotropic_hardening.empty())
    {
      // se
      d2out_din2->set(
          -4.0 * se / math::pow(sf, 3.0) * I, yield_function, isotropic_hardening, flow_invariant);
      // sp
      d2out_din2->set(-f * _q1 * _q2 *
                          (_q2 * sp * math::cosh(_q2 / 2.0 * sp / sf) +
                           2.0 * sf * math::sinh(_q2 / 2.0 * sp / sf)) /
                          (2.0 * math::pow(sf, 3.0)) * I,
                      yield_function,
                      isotropic_hardening,
                      poro_invariant);
      // f
      d2out_din2->set(-_q1 * _q2 * sp * math::sinh(_q2 / 2.0 * sp / sf) / math::pow(sf, 2.0) * I,
                      yield_function,
                      isotropic_hardening,
                      void_fraction);
      // isotropic hardening
      d2out_din2->set((12 * math::pow(se, 2.0) + f * _q1 * _q2 * sp *
                                                     (_q2 * sp * math::cosh(_q2 / 2.0 * sp / sf) +
                                                      4.0 * sf * math::sinh(_q2 / 2.0 * sp / sf))) /
                          (2 * math::pow(sf, 4.0)) * I,
                      yield_function,
                      isotropic_hardening,
                      isotropic_hardening);
      // sy
      if (has_nonlinear_parameter("sy"))
        d2out_din2->set(
            (12 * math::pow(se, 2.0) + f * _q1 * _q2 * sp *
                                           (_q2 * sp * math::cosh(_q2 / 2.0 * sp / sf) +
                                            4.0 * sf * math::sinh(_q2 / 2.0 * sp / sf))) /
                (2 * math::pow(sf, 4.0)) * I,
            yield_function,
            isotropic_hardening,
            nl_param("sy"));
      // q1
      if (has_nonlinear_parameter("q1"))
        d2out_din2->set(-f * _q2 * sp * math::sinh(_q2 / 2.0 * sp / sf) / math::pow(sf, 2.0) * I,
                        yield_function,
                        isotropic_hardening,
                        nl_param("q1"));
      // q2
      if (has_nonlinear_parameter("q2"))
        d2out_din2->set(-f * _q1 * sp *
                            (_q2 * sp * math::cosh(_q2 / 2.0 * sp / sf) +
                             2.0 * sf * math::sinh(_q2 / 2.0 * sp / sf)) /
                            (2 * math::pow(sf, 3.0)) * I,
                        yield_function,
                        isotropic_hardening,
                        nl_param("q2"));
    }

    // sy
    if (has_nonlinear_parameter("sy"))
    {
      // se
      d2out_din2->set(
          -4.0 * se / math::pow(sf, 3.0) * I, yield_function, nl_param("sy"), flow_invariant);
      // sp
      d2out_din2->set(-f * _q1 * _q2 *
                          (_q2 * sp * math::cosh(_q2 / 2.0 * sp / sf) +
                           2.0 * sf * math::sinh(_q2 / 2.0 * sp / sf)) /
                          (2.0 * math::pow(sf, 3.0)) * I,
                      yield_function,
                      isotropic_hardening,
                      poro_invariant);
      // f
      d2out_din2->set(-_q1 * _q2 * sp * math::sinh(_q2 / 2.0 * sp / sf) / math::pow(sf, 2.0) * I,
                      yield_function,
                      nl_param("sy"),
                      void_fraction);
      // isotropic hardening
      if (!isotropic_hardening.empty())
        d2out_din2->set(
            (12 * math::pow(se, 2.0) + f * _q1 * _q2 * sp *
                                           (_q2 * sp * math::cosh(_q2 / 2.0 * sp / sf) +
                                            4.0 * sf * math::sinh(_q2 / 2.0 * sp / sf))) /
                (2 * math::pow(sf, 4.0)) * I,
            yield_function,
            nl_param("sy"),
            isotropic_hardening);
      // sy
      d2out_din2->set((12 * math::pow(se, 2.0) + f * _q1 * _q2 * sp *
                                                     (_q2 * sp * math::cosh(_q2 / 2.0 * sp / sf) +
                                                      4.0 * sf * math::sinh(_q2 / 2.0 * sp / sf))) /
                          (2 * math::pow(sf, 4.0)) * I,
                      yield_function,
                      nl_param("sy"),
                      nl_param("sy"));
      // q1
      if (has_nonlinear_parameter("q1"))
        d2out_din2->set(-f * _q2 * sp * math::sinh(_q2 / 2.0 * sp / sf) / math::pow(sf, 2.0) * I,
                        yield_function,
                        nl_param("sy"),
                        nl_param("q1"));
      // q2
      if (has_nonlinear_parameter("q2"))
        d2out_din2->set(-f * _q1 * sp *
                            (_q2 * sp * math::cosh(_q2 / 2.0 * sp / sf) +
                             2.0 * sf * math::sinh(_q2 / 2.0 * sp / sf)) /
                            (2 * math::pow(sf, 3.0)) * I,
                        yield_function,
                        nl_param("sy"),
                        nl_param("q2"));
    }

    // q1
    if (has_nonlinear_parameter("q1"))
    {
      // poro invariant
      d2out_din2->set(f * _q2 * math::sinh(_q2 / 2.0 * sp / sf) / sf * I,
                      yield_function,
                      nl_param("q1"),
                      poro_invariant);
      // void fraction
      d2out_din2->set(
          2.0 * math::cosh(_q2 / 2.0 * sp / sf) * I, yield_function, nl_param("q1"), void_fraction);

      // isotropic hardening
      if (!isotropic_hardening.empty())
        d2out_din2->set(-f * _q2 * sp * math::sinh(_q2 / 2.0 * sp / sf) / math::pow(sf, 2.0) * I,
                        yield_function,
                        nl_param("q1"),
                        isotropic_hardening);
      // sy
      if (has_nonlinear_parameter("sy"))
        d2out_din2->set(-f * _q2 * sp * math::sinh(_q2 / 2.0 * sp / sf) / math::pow(sf, 2.0) * I,
                        yield_function,
                        nl_param("q1"),
                        nl_param("sy"));
      // q2
      if (has_nonlinear_parameter("q2"))
        d2out_din2->set(f * sp * math::sinh(_q2 / 2.0 * sp / sf) / sf * I,
                        yield_function,
                        nl_param("q1"),
                        nl_param("q2"));
    }

    // q2
    if (has_nonlinear_parameter("q2"))
    {
      // poro invariant
      d2out_din2->set(f * _q1 *
                          (_q2 * sp * math::cosh(_q2 / 2.0 * sp / sf) +
                           2 * sf * math::sinh(_q2 / 2.0 * sp / sf)) /
                          (2 * math::pow(sf, 2.0)) * I,
                      yield_function,
                      nl_param("q2"),
                      poro_invariant);
      // void fraction
      d2out_din2->set(_q1 * sp * math::sinh(_q2 / 2.0 * sp / sf) / sf * I,
                      yield_function,
                      nl_param("q2"),
                      void_fraction);
      // isotropic hardening
      if (!isotropic_hardening.empty())
        d2out_din2->set(-f * _q1 * sp *
                            (_q2 * sp * math::cosh(_q2 / 2.0 * sp / sf) +
                             2 * sf * math::sinh(_q2 / 2.0 * sp / sf)) /
                            (2 * math::pow(sf, 3.0)) * I,
                        yield_function,
                        nl_param("q2"),
                        isotropic_hardening);
      // sy
      if (has_nonlinear_parameter("sy"))
        d2out_din2->set(-f * _q1 * sp *
                            (_q2 * sp * math::cosh(_q2 / 2.0 * sp / sf) +
                             2 * sf * math::sinh(_q2 / 2.0 * sp / sf)) /
                            (2 * math::pow(sf, 3.0)) * I,
                        yield_function,
                        nl_param("q2"),
                        nl_param("sy"));
      // q1
      if (has_nonlinear_parameter("q1"))
        d2out_din2->set(f * sp * math::sinh(_q2 / 2.0 * sp / sf) / sf * I,
                        yield_function,
                        nl_param("q2"),
                        nl_param("q1"));
      // q2
      if (has_nonlinear_parameter("q2"))
        d2out_din2->set(f * _q1 * math::pow(sp, 2.0) * math::cosh(_q2 / 2.0 * sp / sf) /
                            (2 * math::pow(sf, 2.0)) * I,
                        yield_function,
                        nl_param("q2"),
                        nl_param("q2"));
    }

    // q3
    if (has_nonlinear_parameter("q3"))
    {
      // void fraction
      d2out_din2->set(-2.0 * f * I, yield_function, nl_param("q3"), void_fraction);
    }
  }
}
} // namespace neml2
