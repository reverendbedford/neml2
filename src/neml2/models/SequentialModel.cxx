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

#include "neml2/models/SequentialModel.h"

namespace neml2
{
register_NEML2_object(SequentialModel);

OptionSet
SequentialModel::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<std::vector<std::string>>("models");
  return options;
}

SequentialModel::SequentialModel(const OptionSet & options)
  : Model(options)
{
  // Register all the models, merging input
  for (const auto & model_name : options.get<std::vector<std::string>>("models"))
    register_model<Model>(model_name, 0, /*nonlinear=*/false, /*merge_input=*/false);

  // Input is the merged input of all the models
  for (auto & model : registered_models())
    for (auto && [name, var] : model->output_views())
      if (!input_axis().has_variable(name))
        declare_input_variable(var.base_storage(), name);

  // Output is the merged output of all the models, except here we need to throw if a variable is
  // repeated
  for (auto & model : registered_models())
    for (auto && [name, var] : model->output_views())
      if (output_axis().has_variable(name))
        throw NEMLException("Two submodels in a SequentialModel declare the same output.");
      else
        declare_output_variable(var.base_storage(), name);

  std::cout << "A" << std::endl;
  for (auto && [name, var] : input_views())
    std::cout << name << std::endl;

  std::cout << "B" << std::endl;
  for (auto && [name, var] : output_views())
    std::cout << name << std::endl;
}

void
SequentialModel::setup_submodel_input_views()
{
  for (auto submodel : registered_models())
  {
  }
}

void
SequentialModel::set_value(bool out, bool dout_din, bool d2out_din2)
{
  (void)out;
  (void)dout_din;
  (void)d2out_din2;
}

} // namespace neml2
