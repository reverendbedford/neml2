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

#include <future>

namespace neml2
{
class ComposedModel : public Model
{
public:
  static OptionSet expected_options();

  ComposedModel(const OptionSet & options);

  virtual void check_AD_limitation() const override;

  virtual std::map<std::string, const VariableBase *>
  named_nonlinear_parameters(bool recursive = false) const override;

  virtual std::map<std::string, Model *>
  named_nonlinear_parameter_models(bool recursive = false) const override;

  virtual void setup() override;

protected:
  virtual void allocate_variables(bool in, bool out) override;

  /**
   * Setup each of the sub-model's input views. Note the logic is different from the base class's.
   * The sub-models in a composed model should not all view into the host's input storage. Instead,
   * the non-inbound input variables should view into dependent models' output storage. This avoids
   * all the copying when passing a sub-model's output as another sub-model's input.
   *
   */
  virtual void setup_submodel_input_views(VariableStore * host) override;

  virtual void setup_output_views() override;

  void set_value(bool, bool, bool) override;

private:
  // Custom comparator for sorting assembly indices
  struct SliceCmp
  {
    bool operator()(const indexing::TensorIndex & a, const indexing::TensorIndex & b) const
    {
      neml_assert(a.is_slice() && b.is_slice(), "Comparator must be used on slices");
      neml_assert(a.slice().step().expect_int() == 1 && b.slice().step().expect_int() == 1,
                  "Slices must have step == 1");
      return a.slice().start().expect_int() < b.slice().start().expect_int();
    }
  };

  /// Helper method to evaluate one single model in the threaded set_value loop
  void set_value_async(Model * i, bool out, bool dout_din, bool d2out_din2);

  /// Helper method to recursively apply chain rule
  void apply_chain_rule(Model * model);

  /// Helper method to recursively apply second order chain rule
  void apply_second_order_chain_rule(Model * model);

  /// Additional outbound items in the dependency graph
  const std::vector<VariableName> _additional_outputs;

  /// Whether to automatically add nonlinear parameters
  const bool _auto_nl_param;

  /// Helper to resolve model dependency
  DependencyResolver<Model, VariableName> _dependency;

  /// Assembly indices
  std::map<Model *,
           std::map<indexing::TensorIndex, std::pair<Model *, indexing::TensorIndex>, SliceCmp>>
      _assembly_indices;

  /// Cache for partial derivatives of model inputs w.r.t. total input
  std::map<Model *, std::vector<BatchTensor>> _dpin_din_views;

  /// Cache for partial derivatives of model outputs w.r.t. total input
  std::map<Model *, LabeledMatrix> _dpout_din;

  /// Cache for second partial derivatives of model inputs w.r.t. total input
  std::map<Model *, std::vector<BatchTensor>> _d2pin_din2_views;

  /// Cache for second partial derivatives of model outputs w.r.t. total input
  std::map<Model *, LabeledTensor3D> _d2pout_din2;

  /// Threaded evaluation results of sub-models
  std::map<Model *, std::future<void>> _async_results;
};
} // namespace neml2
