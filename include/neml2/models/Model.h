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

#include "neml2/base/DependencyDefinition.h"

#include "neml2/models/Data.h"
#include "neml2/models/ParameterStore.h"
#include "neml2/models/VariableStore.h"
#include "neml2/solvers/NonlinearSystem.h"

#include "neml2/tensors/LabeledVector.h"
#include "neml2/tensors/LabeledMatrix.h"
#include "neml2/tensors/LabeledTensor3D.h"

namespace neml2
{
/**
 * @brief The base class for all constitutive models.
 *
 * A model maps some input to output. The forward operator (and its derivative) is defined in the
 * method \p set_value. All concrete models must provide the implementation of the forward operator
 * by overriding the \p set_value method.
 */
class Model : public std::enable_shared_from_this<Model>,
              public Data,
              public ParameterStore,
              public VariableStore,
              public NonlinearSystem,
              public DependencyDefinition<VariableName>
{
public:
  static OptionSet expected_options();

  /**
   * @brief Construct a new Model object
   *
   * @param options The options extracted from the input file
   */
  Model(const OptionSet & options);

  /**
   * @brief Check for common problems
   *
   * This method serves as the entry point for diagnosing common problems in model setup,
   * composition, etc.
   *
   * @returns A vector of exceptions of type Diagnosis for each of the detected problem.
   */
  virtual std::vector<Diagnosis> preflight() const;

  /// Whether this model defines one or more nonlinear equations to be solved
  virtual bool is_nonlinear_system() const { return _nonlinear_system; }

  /// Whether inferece mode is on
  bool inference_mode() const { return _inference_mode; }

  /**
   * @brief Allocate storage and setup views for all the variables of this model and recursively all
   * of the sub-models. See the other overload for detailed description.
   */
  virtual void reinit(const Tensor & tensor, int deriv_order);

  /**
   * @brief Allocate storage and setup views for all the variables of this model and recursively all
   * of the sub-models.
   *
   * This method must be called before any call to the forward operators, e.g., value, dvalue,
   * value_and_dvalue, etc.
   *
   * IMPORTANT: If the batch shape of this model changes, this method must be called again to
   * re-allocate the storage and views.
   *
   * @param batch_shape Batch shape of the input, output and derivatives
   * @param deriv_order Order of derivative required for this model
   * @param device Device on which the model will be evaluated
   * @param dtype Number type, e.g., torch::kFloat32, torch::kFloat64, etc
   */
  virtual void reinit(TensorShapeRef batch_shape,
                      int deriv_order = 0,
                      const torch::Device & device = default_device(),
                      const torch::Dtype & dtype = default_dtype());

  /// Prepare for evaluation
  void prepare();

  /// Whether derivative has been requested for this model
  bool requires_grad() const { return _deriv_order >= 1; }

  /// Whether 2nd derivative has been requested for this model
  bool requires_2nd_grad() const { return _deriv_order >= 2; }

  /// This model's batch dim
  Size batch_dim() const { return _batch_sizes.size(); }

  /// This model's batch shape
  TensorShapeRef batch_sizes() const { return _batch_sizes; }

  /// This model's tensor options
  const torch::TensorOptions & options() const { return _options; }

  /// The models that may be used during the evaluation of this model
  const std::vector<Model *> & registered_models() const { return _registered_models; }

  /// Get a registered model by its name
  Model * registered_model(const std::string & name) const;

  /// The variables that this model depends on
  virtual const std::set<VariableName> consumed_items() const override;

  /// The variables that this model defines as part of its output
  virtual const std::set<VariableName> provided_items() const override;

  /**
   * Validate the currently requested AD settings.
   *  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   *  AD_1st_deriv   AD_2nd_deriv   comment
   *          true           true   okay, just slow
   *          true          false   error, this is a weird case
   *         false           true   okay
   *         false          false   great, everything handcoded
   *  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   */
  virtual void check_AD_limitation() const;

  /// Set requires_grad for the input variables
  void input_requires_grad_(bool req = true);

  /// Whether this model is using AD to get 1st derivatives
  bool using_AD_1st_derivative() const { return _AD_1st_deriv; }

  /// Whether this model is using AD to get 2nd derivatives
  bool using_AD_2nd_derivative() const { return _AD_2nd_deriv; }

  /// Tell this model to use AD to get derivatives
  void use_AD_derivatives(bool first = true, bool second = true);

  /// Set \p in to be the input of this model
  virtual void set_input(const LabeledVector & in);

  /// \return the output of this model
  virtual LabeledVector get_output();

  /// \return the derivative of the output w.r.t. the input of this model
  virtual LabeledMatrix get_doutput_dinput();

  /// \return the second derivative of the output w.r.t. the input of this model
  virtual LabeledTensor3D get_d2output_dinput2();

  /// Convenient shortcut to construct and return the model value
  virtual LabeledVector value(const LabeledVector & in);

  /// Convenient shortcut to construct and return the model value and its derivative
  virtual std::tuple<LabeledVector, LabeledMatrix> value_and_dvalue(const LabeledVector & in);

  /// Convenient shortcut to construct and return the model's value, first and second derivative
  virtual std::tuple<LabeledVector, LabeledMatrix, LabeledTensor3D>
  value_and_dvalue_and_d2value(const LabeledVector & in);

  virtual void value();
  virtual void value_and_dvalue();
  virtual void value_and_dvalue_and_d2value();

  /// Check for potential in-place operation
  void check_inplace_dbg();

  /**
   * A model can be treated as an implicit model. An implicit model need to be "solved": the state
   * variables should be iteratively updated until the residual becomes zero. During the SOLVING
   * stage, we only need the derivative of output with respect to the input state. During the
   * UPDATING stage, we only need the derivative of output with respect to the input forces, old
   * forces, and old state. Therefore, the model can/should avoid unnecessary computations by
   * examining the current `stage`.
   */
  static enum Stage { SOLVING, UPDATING } stage;

  /// Declaration of nonlinear parameters may require manipulation of input
  friend class ParameterStore;

  /// ComposedModel's set_value need to call submodel's set_value
  friend class ComposedModel;

protected:
  /**
   * @brief Setup this model.
   * 1. Setup the layout of the input and output axes.
   * 2. Setup the arguments of each variable.
   */
  virtual void setup() override;

  using VariableStore::allocate_variables;

  /// Call VariableStore::allocate_variables recursively on all submodels
  virtual void allocate_variables(bool in, bool out);

  /// Call VariableStore::setup_input_views recursively on all submodels
  virtual void setup_input_views(VariableStore * host = nullptr) override;
  virtual void setup_submodel_input_views(VariableStore * host);

  /// Call VariableStore::setup_output_views recursively on all submodels
  using VariableStore::setup_output_views;
  virtual void setup_output_views();
  virtual void setup_submodel_output_views();

  virtual void setup_nonlinear_system();

  /**
   * @brief Allocate storage and setup views for all the variables of this model and recursively all
   * of the sub-models. See the other overload for detailed description.
   */
  virtual void reinit(bool in, bool out);

  /// Call VariableStore::zero recursively on all submodels
  using VariableStore::zero;
  virtual void zero();

  /// Set \p x as the current solution of the nonlinear system
  virtual void set_solution(const Tensor & x) override;

  /// The map between input -> output, and optionally its derivatives
  virtual void set_value(bool out, bool dout_din, bool d2out_din2) = 0;

  using VariableStore::cache;
  virtual void cache(TensorShapeRef batch_shape,
                     int deriv_order,
                     const torch::Device & device,
                     const torch::Dtype & dtype);

  /**
   * @brief Register a model that the current model may use during its evaluation.
   *
   * If \p merge_input is set to true, this model will also *consume* the consumed variables of \p
   * model, which will affect dependency resolution inside a ComposedModel.
   *
   * @param name The model to register
   * @param extra_deriv_order The additional derivative order required for the registered-submodel
   * @param nonlinear Set to true if the registered model defines a nonlinear system to be solved
   * @param merge_input Whether to merge the input of the registered model into *this* model's
   * input.
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<Model, T>>>
  T & register_model(const std::string & name,
                     int extra_deriv_order = 0,
                     bool nonlinear = false,
                     bool merge_input = true)
  {
    OptionSet extra_opts;
    extra_opts.set<NEML2Object *>("_host") = host();
    extra_opts.set<int>("_extra_derivative_order") = extra_deriv_order;
    extra_opts.set<bool>("_nonlinear_system") = nonlinear;
    extra_opts.set<bool>("_inference_mode") = input_options().get<bool>("_inference_mode");

    auto model = Factory::get_object_ptr<Model>("Models", name, extra_opts);

    if (merge_input)
      for (auto && [name, var] : model->input_views())
        declare_input_variable(var.base_storage(), var.type(), name);

    _registered_models.push_back(model.get());
    return *(std::dynamic_pointer_cast<T>(model));
  }

  virtual void assemble(bool residual, bool Jacobian) override;

  /// Models *this* model may use during its evaluation
  std::vector<Model *> _registered_models;

  /// Whether to use AD to compute 1st derivatives
  bool _AD_1st_deriv;

  /// Whether to use AD to compute 2nd derivatives
  bool _AD_2nd_deriv;

private:
  /// Helper method to extract derivatives after back propagation
  void extract_derivatives(bool retain_graph, bool create_graph, bool allow_unused);

  /// Helper method to extract second derivatives after back propagation
  void extract_second_derivatives(bool retain_graph, bool create_graph, bool allow_unused);

  /// This model's batch shape
  TensorShape _batch_sizes;

  /// This model's tensor options
  torch::TensorOptions _options;

  /**
   * @brief The derivative order required for this model
   *
   * The input/output, derivative, and second derivative storages will be allocated accordingly when
   * allocate_variables is called.
   *
   * 0: Only value is required
   * 1: First order derivative is required
   * 2: Second order derivative is required
   */
  int _deriv_order;

  /**
   * @brief The extra derivative order required for this model
   *
   * When a model is registered as a sub-model, the parent model may require additional derivative
   * order from the sub-model. For example, some models may define its own outputs as functions of
   * the sub-model's derivatives. Normality is an example of such model.
   */
  const int _extra_deriv_order;

  /// Whether this is a nonlinear system
  bool _nonlinear_system;

#ifndef NDEBUG
  /// Whether this model has been evaluated in the current forward pass
  bool _evaluated_once;
#endif

  /// Whether the evaluation uses inference mode
  const bool _inference_mode;
};
} // namespace neml2
