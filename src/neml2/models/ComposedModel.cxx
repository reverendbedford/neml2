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

#include "neml2/models/ComposedModel.h"

namespace neml2
{
register_NEML2_object(ComposedModel);

OptionSet
ComposedModel::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<std::vector<std::string>>("models");
  return options;
}

ComposedModel::ComposedModel(const OptionSet & options)
  : Model(options)
{
  std::vector<std::shared_ptr<Model>> models;
  for (const auto & model_name : options.get<std::vector<std::string>>("models"))
    models.push_back(Factory::get_object_ptr<Model>("Models", model_name));

  register_dependency(models);

  resolve_dependency();

  // Register the models that are needed for evaluation as submodules
  for (auto i : _evaluation_order)
    register_model(i, /*merge_input=*/false);

  setup();
}

bool
ComposedModel::implicit() const
{
  for (auto i : _evaluation_order)
    if (i->implicit())
      return true;
  return false;
}

void
ComposedModel::register_dependency(const std::vector<std::shared_ptr<Model>> & models)
{
  for (const auto & modeli : models)
  {
    // see which model _provides_ the consumed variables
    for (const auto & consumed_var : modeli->consumed_variables())
    {
      bool provided = false;
      for (const auto & modelj : models)
      {
        // No self dependency
        if (modeli == modelj)
          continue;

        const auto & provided_vars = modelj->provided_variables();
        if (provided_vars.find(consumed_var) != provided_vars.end())
        {
          add_edge(modelj, modeli);
          provided = true;
          break;
        }
      }
      if (!provided)
      {
        _consumed_vars.insert(consumed_var);
        input().add(consumed_var, modeli->input().storage_size(consumed_var));
      }
    }

    // see which model _consumes_ the provided variables
    for (const auto & provided_var : modeli->provided_variables())
    {
      bool consumed = false;
      for (const auto & modelj : models)
      {
        // No self dependency
        if (modeli == modelj)
          continue;

        const auto & consumed_vars = modelj->consumed_variables();
        if (consumed_vars.find(provided_var) != consumed_vars.end())
        {
          consumed = true;
          break;
        }
      }
      if (!consumed)
      {
        _provided_vars.insert(provided_var);
        output().add(provided_var, modeli->output().storage_size(provided_var));
      }
    }

    // Each model may request additional outputs
    for (const auto & var : modeli->additional_outputs())
      output().add(var, modeli->output().storage_size(var));
  }
}

void
ComposedModel::add_edge(const std::shared_ptr<Model> & from, const std::shared_ptr<Model> & to)
{
  add_node(from);
  add_node(to);
  _dependecies[to->name()].push_back(from);
}

void
ComposedModel::add_node(const std::shared_ptr<Model> & model)
{
  _models[model->name()] = model;
  if (_dependecies.count(model->name()) == 0)
    _dependecies[model->name()];
}

void
ComposedModel::resolve_dependency()
{
  // Find the root model(s)
  // Basic idea: if a model is not needed by any other model, then it must be a root model
  std::map<std::string, bool> visited;
  for (const auto & [name, deps] : _dependecies)
    for (auto dep : deps)
      visited[dep->name()] = true;
  for (const auto & [name, i] : _models)
    if (!visited[name])
      _output_models.push_back(i);

  // Figure out the evaluation order
  visited.clear();
  _evaluation_order.clear();
  for (auto i : _output_models)
    resolve_dependency(i, _evaluation_order, visited);
}

void
ComposedModel::resolve_dependency(const std::shared_ptr<Model> & i,
                                  std::vector<std::shared_ptr<Model>> & order,
                                  std::map<std::string, bool> & visited)
{
  // Mark the current node as visited
  visited[i->name()] = true;

  // Recurse for all the dependent models
  for (auto dep : dependent_models(i->name()))
    if (!visited[dep->name()])
      resolve_dependency(dep, order, visited);

  order.push_back(i);
}

void
ComposedModel::set_value(const LabeledVector & in,
                         LabeledVector * out,
                         LabeledMatrix * dout_din,
                         LabeledTensor3D * d2out_din2) const
{
  // pin stands for partial input
  // pout stands for partial output
  const auto batch_sz = in.batch_sizes();
  const auto options = in.options();

  // If only out is requested, we just evaluate all the models following the sorted evaluation
  // order. That's it!
  if (out && !dout_din && !d2out_din2)
  {
    std::map<std::string, LabeledVector> cached_pout;

    // Follow the (sorted) evaluation order to evaluate all the models
    for (auto i : _evaluation_order)
    {
      auto pin = LabeledVector::zeros(batch_sz, {&i->input()}, options);
      pin.fill(in);

      // All the dependencies must have been cached, as we have sorted out the evaluation order
      // according to the dependecies. If a dependency's output hasn't been cached, we screwed up
      // the dependency resolution.
      const auto & deps = dependent_models(i->name());
      for (auto dep : deps)
      {
        neml_assert_dbg(cached_pout.count(dep->name()) > 0,
                        "Internal error, incorrect evaluation order");
        pin.fill(cached_pout.at(dep->name()));
      }

      auto pout = i->value(pin);
      out->fill(pout);

      cached_pout.emplace(i->name(), pout);
    }
    return;
  }
  // If dout_din is requested, we will need to evaluate both the models' value and derivatives, as
  // well as apply first order chain rule to compute the first order total derivatives.
  else if (dout_din && !d2out_din2)
  {
    std::map<std::string, LabeledVector> cached_pout;
    std::map<std::string, LabeledMatrix> cached_dpout_din;

    auto din_din = LabeledMatrix::identity(batch_sz, input(), options);

    // Follow the (sorted) evaluation order to evaluate all the models
    for (auto i : _evaluation_order)
    {
      auto pin = LabeledVector::zeros(batch_sz, {&i->input()}, options);
      auto dpin_din = LabeledMatrix::zeros(batch_sz, {&i->input(), &input()}, options);
      pin.fill(in);
      dpin_din.fill(din_din);

      // All the dependencies must have been cached, as we have sorted out the evaluation order
      // according to the dependecies. If a dependency's output hasn't been cached, we screwed up
      // the dependency resolution.
      const auto & deps = dependent_models(i->name());
      for (auto dep : deps)
      {
        neml_assert_dbg(cached_pout.count(dep->name()) > 0,
                        "Internal error, incorrect evaluation order");
        pin.fill(cached_pout.at(dep->name()));
        dpin_din.fill(cached_dpout_din.at(dep->name()));
      }

      auto [pout, dpout_dpin] = i->value_and_dvalue(pin);
      auto dpout_din = dpout_dpin.chain(dpin_din);

      if (out)
        out->fill(pout);

      dout_din->fill(dpout_din);

      cached_pout.emplace(i->name(), pout);
      cached_dpout_din.emplace(i->name(), dpout_din);
    }
    return;
  }
  // Here's the most sophisticated case: d2out_din2 is requested. In this case each model's value,
  // first and second derivatives are necessary. We need to apply the first order chain rule to
  // compute the first order total derivatives, and the second order chain rule to compute the
  // second order total derivatives.
  else if (d2out_din2)
  {
    std::map<std::string, LabeledVector> cached_pout;
    std::map<std::string, LabeledMatrix> cached_dpout_din;
    std::map<std::string, LabeledTensor3D> cached_d2pout_din2;

    auto din_din = LabeledMatrix::identity(batch_sz, input(), options);

    // Follow the (sorted) evaluation order to evaluate all the models
    for (auto i : _evaluation_order)
    {
      auto pin = LabeledVector::zeros(batch_sz, {&i->input()}, options);
      auto dpin_din = LabeledMatrix::zeros(batch_sz, {&i->input(), &input()}, options);
      auto d2pin_din2 =
          LabeledTensor3D::zeros(batch_sz, {&i->input(), &input(), &input()}, options);
      pin.fill(in);
      dpin_din.fill(din_din);

      // All the dependencies must have been cached, as we have sorted out the evaluation order
      // according to the dependecies. If a dependency's output hasn't been cached, we screwed up
      // the dependency resolution.
      const auto & deps = dependent_models(i->name());
      for (auto dep : deps)
      {
        neml_assert_dbg(cached_pout.count(dep->name()) > 0,
                        "Internal error, incorrect evaluation order");
        pin.fill(cached_pout.at(dep->name()));
        dpin_din.fill(cached_dpout_din.at(dep->name()));
        d2pin_din2.fill(cached_d2pout_din2.at(dep->name()));
      }

      auto [pout, dpout_dpin, d2pout_dpin2] = i->value_and_dvalue_and_d2value(pin);
      auto dpout_din = dpout_dpin.chain(dpin_din);
      auto d2pout_din2 = d2pout_dpin2.chain(d2pin_din2, dpout_dpin, dpin_din);

      if (out)
        out->fill(pout);

      if (dout_din)
        dout_din->fill(dpout_din);

      d2out_din2->fill(d2pout_din2);

      cached_pout.emplace(i->name(), pout);
      cached_dpout_din.emplace(i->name(), dpout_din);
      cached_d2pout_din2.emplace(i->name(), d2pout_din2);
    }
  }
  else
    throw NEMLException("Logic error");
}

const std::vector<std::shared_ptr<Model>> &
ComposedModel::dependent_models(const std::string & n) const
{
  neml_assert(_models.count(n) != 0, n, " is not registered in ", name());

  return _dependecies.at(n);
}

// LCOV_EXCL_START
std::string
ComposedModel::evaluation_order() const
{
  std::stringstream ss;
  for (auto i : _evaluation_order)
    ss << i->name() << " ";
  return ss.str();
}
// LCOV_EXCL_STOP

void
ComposedModel::to_dot(std::ostream & os) const
{
  // Keep track of input output IDs so that I can connect them later
  std::map<std::string, int> io_ids;

  // Preemble
  int id = 0;
  os << "digraph {\n";
  os << "compound = true\n";
  os << "graph [ranksep = 4, penwidth = 2]\n";

  // Write all the models
  for (auto i : _evaluation_order)
    to_dot(os, *i, id, io_ids);

  os << "}\n";
}

void
ComposedModel::to_dot(std::ostream & os,
                      const Model & model,
                      int & id,
                      std::map<std::string, int> & io_ids) const
{
  // Preemble
  os << "subgraph ";
  os << "cluster_" << id++ << " ";
  os << "{\n";
  os << "label = \"" << model.name() << "\"\n";

  io_ids.emplace(model.name() + " input", id);
  input().to_dot(os, id, model.name() + " input", true, true);
  io_ids.emplace(model.name() + " output", id);
  output().to_dot(os, id, model.name() + " output", true, true);

  for (auto i : dependent_models(model.name()))
  {
    os << "\"" << i->name() + " output\"";
    os << " -> ";
    os << "\"" << model.name() << " input\"";
    os << "[ltail = cluster_" << io_ids[i->name() + " output"] << ", ";
    os << "lhead = cluster_" << io_ids[model.name() + " input"] << ", penwidth = 2]\n";
  }

  os << "}\n";
}

// LCOV_EXCL_START
void
ComposedModel::print_dependency(std::ostream & os) const
{
  for (const auto & [name, deps] : _dependecies)
  {
    os << name << " depends on {";
    for (const auto & dep : deps)
      os << dep->name() << ", ";
    os << "}" << std::endl;
  }
  if (_dependecies.empty())
    os << std::endl;
}
// LCOV_EXCL_STOP
} // namespace neml2
