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
ComposedModel::ComposedModel(const std::string & name,
                             const std::vector<std::shared_ptr<Model>> & models,
                             const std::vector<LabeledAxisAccessor> & additional_outputs)
  : Model(name)
{
  register_dependency(models);

  for (const auto & var : additional_outputs)
    _provided_vars.insert(var);

  // Set up the input axis
  input().clear();
  for (const auto & var : _consumed_vars)
    input().add(var);

  // Set up the output axis
  output().clear();
  for (const auto & var : _provided_vars)
    output().add(var);

  // Find the root model(s)
  // Basic idea: if a model is not needed by any other model, then it must be a root model
  std::map<std::string, bool> visited;
  for (const auto & [name, deps] : _dependecies)
    for (auto dep : deps)
      visited[dep->name()] = true;
  for (const auto & [name, i] : _models)
    if (!visited[name])
      _output_models.push_back(i);

  resolve_dependency();

  // Register the models that are needed for evaluation as submodules
  for (auto i : _evaluation_order)
    register_module(i->name(), i);

  setup();
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
        _consumed_vars.insert(consumed_var);
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
        _provided_vars.insert(provided_var);
    }
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
  // Figure out the evaluation order
  std::map<std::string, bool> visited;
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
ComposedModel::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  // pin stands for partial input
  // pout stands for partial output
  TorchSize nbatch = in.batch_size();

  if (dout_din)
  {
    std::map<std::string, LabeledVector> cached_pout;
    std::map<std::string, LabeledMatrix> cached_dpout_din;

    auto din_din = LabeledMatrix::identity(nbatch, input());

    // Follow the (sorted) evaluation order to evaluate all the models
    for (auto i : _evaluation_order)
    {
      LabeledVector pin(nbatch, i->input());
      LabeledMatrix dpin_din(nbatch, i->input(), input());
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

      out.fill(pout);
      dout_din->fill(dpout_din);

      cached_pout.emplace(i->name(), pout);
      cached_dpout_din.emplace(i->name(), dpout_din);
    }
  }
  else
  {
    std::map<std::string, LabeledVector> cached_pout;

    // Follow the (sorted) evaluation order to evaluate all the models
    for (auto i : _evaluation_order)
    {
      LabeledVector pin(nbatch, i->input());
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
      out.fill(pout);

      cached_pout.emplace(i->name(), pout);
    }
  }
}

const std::vector<std::shared_ptr<Model>> &
ComposedModel::dependent_models(const std::string & n) const
{
  neml_assert(_models.count(n) != 0, n, " is not registered in ", name());

  return _dependecies.at(n);
}

std::string
ComposedModel::evaluation_order() const
{
  std::stringstream ss;
  for (auto i : _evaluation_order)
    ss << i->name() << " ";
  return ss.str();
}

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
} // namespace neml2
