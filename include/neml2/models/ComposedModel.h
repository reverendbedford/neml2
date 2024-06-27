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
#include "neml2/base/DependencyResolver.h"

namespace neml2
{
class ComposedModel : public Model
{
public:
  static OptionSet expected_options();

  ComposedModel(const OptionSet & options);

  virtual void check_AD_limitation() const override;

protected:
  /// Recursively register sub-model's nonlinar parameters
  void register_nonlinear_params(Model & m);

  virtual void allocate_variables(int deriv_order, bool options_changed) override;

  /**
   * Setup each of the sub-model's input views. Note the logic is different from the base class's.
   * The sub-models in a composed model should not all view into the host's input storage. Instead,
   * the non-inbound input variables should view into dependent models' output storage. This avoids
   * all the copying when passing a sub-model's output as another sub-model's input.
   *
   */
  virtual void setup_submodel_input_views() override;

  void set_value(bool, bool, bool) override;

private:
  void clear_chain_rule_cache()
  {
    _dpout_din.clear();
    _d2pout_din2.clear();
  }

  /// Helper method to recursively apply chain rule
  void apply_chain_rule(Model * model);

  /// Helper method to recursively apply second order chain rule
  void apply_second_order_chain_rule(Model * model);

  /// Additional outbound items in the dependency graph
  const std::vector<VariableName> _additional_outputs;

  /// Helper to resolve model dependency
  DependencyResolver<Model, VariableName> _dependency;

  /// Starting point of chain rule
  LabeledMatrix _din_din;

  /// Cache for partial derivatives of model outputs w.r.t. total input
  std::map<Model *, LabeledMatrix> _dpout_din;

  /// Cache for second partial derivatives of model outputs w.r.t. total input
  std::map<Model *, LabeledTensor3D> _d2pout_din2;
};
} // namespace neml2
