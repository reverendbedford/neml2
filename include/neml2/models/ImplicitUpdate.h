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

#pragma once

#include "neml2/models/Model.h"
#include "neml2/solvers/NonlinearSolver.h"
#include "neml2/predictors/Predictor.h"

namespace neml2
{
class ImplicitUpdate : public Model
{
public:
  static ParameterSet expected_params();

  ImplicitUpdate(const ParameterSet & name);

  const Model & implicit_model() const { return _model; }

  virtual void advance_step() override;

  // dvalue is overriden because we rely on the implicit function theorem to compute the
  // derivatives, and so the value has to be evaluated even if only the derivatives are requested.
  virtual LabeledMatrix dvalue(const LabeledVector & in) const override;

protected:
  virtual void set_value(const LabeledVector & in,
                         LabeledVector * out,
                         LabeledMatrix * dout_din = nullptr,
                         LabeledTensor3D * d2out_din2 = nullptr) const override;

  /// The implicit model to be updated
  Model & _model;

  /// The nonlinear solver used to solve the nonlinear system
  const NonlinearSolver & _solver;

  /// The predictor used to set the initial guess
  Predictor * _predictor;
};
} // namespace neml2
