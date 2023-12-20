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
class NewModel : public Data,
                 public ParameterStore,
                 public VariableStore,
                 public NonlinearSystem,
                 public DependencyDefinition<LabeledAxisAccessor>
{
public:
  static OptionSet expected_options();

  /**
   * @brief Construct a new NewModel object
   *
   * @param options The options extracted from the input file
   */
  NewModel(const OptionSet & options);

  /// Is this model an implicit system?
  bool implicit() const;

  /**
   * @brief Allocate storage and setup views for all the variables of this model and recursively all
   * of the sub-models.
   *
   * This method must be called before any call to the forward operators, e.g., value, dvalue,
   * value_and_dvalue, etc.
   *
   * IMPORTANT: If the batch shape of this model changes, this method must be called again to
   * re-allocate the storage and views.
   */
  virtual void reinit(TorchShapeRef batch_shape,
                      const torch::TensorOptions & options = default_tensor_options());

  /**
   * @brief Allocate storage and setup views for all the variables of this model and recursively all
   * of the sub-models.
   */
  virtual void reinit(const BatchTensor & tensor);

  /// This model's batch dim
  TorchSize batch_dim() const { return _batch_sizes.size(); }

  /// This model's batch shape
  TorchShapeRef batch_sizes() const { return _batch_sizes; }

  /// This model's tensor options
  const torch::TensorOptions & options() const { return _options; }

  /// The models that may be used during the evaluation of this model
  const std::vector<NewModel *> & registered_models() const { return _registered_models; }

  /// The variables that this model depends on
  virtual const std::set<LabeledAxisAccessor> consumed_items() const override;

  /// The variables that this model defines as part of its output
  virtual const std::set<LabeledAxisAccessor> provided_items() const override;

  /**
   * The additional variables that this model should provide. Typically these variables are not
   * directly computed by this model, instead they come from other information that this model
   * _knows_, e.g., directly from the input variables.
   */
  const std::vector<LabeledAxisAccessor> & additional_outputs() const
  {
    return _additional_outputs;
  }

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

  /// Call VariableStore::allocate_variables recursively on all submodels
  virtual void allocate_variables(TorchShapeRef batch_shape,
                                  const torch::TensorOptions & options) override;

  /// Call VariableStore::setup_input_views recursively on all submodels
  virtual void setup_input_views() override;
  virtual void setup_submodel_input_views();

  /// Call VariableStore::setup_output_views recursively on all submodels
  virtual void setup_output_views(bool out, bool dout_din = true, bool d2out_din2 = true) override;
  virtual void setup_submodel_output_views(bool out, bool dout_din = true, bool d2out_din2 = true);

  /// Call VariableStore::detach_and_zero recursively on all submodels
  virtual void detach_and_zero(bool out, bool dout_din = true, bool d2out_din2 = true) override;

  /// The map between input -> output, and optionally its derivatives
  virtual void set_value(bool out, bool dout_din, bool d2out_din2) = 0;

  virtual void cache(TorchShapeRef batch_shape) override;

  virtual void cache(const torch::TensorOptions & options);

  virtual void reinit_implicit_system() override;

  /**
   * @brief Register a model that the current model may use during its evaluation.
   *
   * If \p merge_input is set to true, this model will also *consume* the consumed variables of \p
   * model, which will affect dependency resolution inside a ComposedModel.
   *
   * @param model The model to register
   * @param merge_input Whether to merge the input of the registered model into *this* model's
   * input.
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<NewModel, T>>>
  T & register_model(const std::string & name, bool merge_input = true)
  {
    auto model = Factory::get_object_ptr<NewModel>("Models", name, host(), /*force_create=*/true);

    if (merge_input)
      for (auto && [name, var] : model->input_views())
        declare_input_variable(name, var.base_storage());

    _registered_models.push_back(model.get());
    return *(std::dynamic_pointer_cast<T>(model));
  }

  virtual void assemble(bool residual, bool Jacobian) override;

  /// NewModels *this* model may use during its evaluation
  std::vector<NewModel *> _registered_models;

  std::vector<LabeledAxisAccessor> _additional_outputs;

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
  TorchShape _batch_sizes;

  /// This model's tensor options
  torch::TensorOptions _options;
};
} // namespace neml2
