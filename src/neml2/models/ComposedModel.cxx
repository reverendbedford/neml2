#include "neml2/models/ComposedModel.h"

namespace neml2
{
ComposedModel::ComposedModel(const std::string & name,
                             const std::vector<std::shared_ptr<Model>> & models,
                             const std::vector<std::shared_ptr<Model>> & input_models,
                             const std::vector<std::shared_ptr<Model>> & output_models)
  : Model(name)
{
  for (const auto & modeli : models)
  {
    // input models don't consume anything
    if (std::find(input_models.begin(), input_models.end(), modeli) != input_models.end())
      continue;

    // see which model _provides_ the consumed variables
    for (const auto & consumed_var : modeli->consumed_variables())
      for (const auto & modelj : models)
      {
        if (modeli == modelj)
          continue;
        // output models don't provide anything
        if (std::find(output_models.begin(), output_models.end(), modelj) != output_models.end())
          continue;
        const auto & provided_vars = modelj->provided_variables();
        if (std::find(provided_vars.begin(), provided_vars.end(), consumed_var) !=
            provided_vars.end())
          register_dependency(modelj, modeli);
      }
  }

  resolve_dependency();

  // Registered the models that are needed for evaluation as submodules
  for (auto i : _evaluation_order)
    register_module(i->name(), i);

  setup();
}

void
ComposedModel::add_node(const std::shared_ptr<Model> & model)
{
  _models[model->name()] = model;
  if (_dependecies.count(model->name()) == 0)
    _dependecies[model->name()];
}

void
ComposedModel::register_dependency(const std::shared_ptr<Model> & from,
                                   const std::shared_ptr<Model> & to)
{
  add_node(from);
  add_node(to);
  _dependecies[to->name()].push_back(from);
}

const std::vector<std::shared_ptr<Model>> &
ComposedModel::dependent_models(const std::string & n) const
{
  if (_models.count(n) == 0)
    throw std::runtime_error(n + " is not registered model in " + name());

  return _dependecies.at(n);
}

void
ComposedModel::set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din) const
{
  // pin stands for partial input
  // pout stands for partial output
  TorchSize nbatch = in.batch_size();
  std::map<std::string, LabeledVector> cached_pout;
  std::map<std::string, LabeledMatrix> cached_dpout_dpin;

  // Follow the (sorted) evaluation order to evaluate all the models
  for (auto i : _evaluation_order)
  {
    LabeledVector pin(nbatch, i->input());
    const auto & deps = dependent_models(i->name());

    // If this is a leaf model, the total input is its input
    if (deps.empty())
      pin = in;
    // Otherwise grab the input from the cached_pout
    else
    {
      // All the dependencies must have been cached, as we have sorted out the evaluation order
      // according to the dependecies. If a dependency's output hasn't been cached, we screwed up
      // the dependency resolution.
      for (auto dep : deps)
      {
        if (cached_pout.count(dep->name()) == 0)
          throw std::runtime_error("Internal error, incorrect evaluation order");
        pin.fill(cached_pout.at(dep->name()));
      }
    }
    if (dout_din)
    {
      auto [pout, dpout_dpin] = i->value_and_dvalue(pin);
      cached_pout.emplace(i->name(), pout);
      cached_dpout_dpin.emplace(i->name(), dpout_dpin);
    }
    else
    {
      auto pout = i->value(pin);
      cached_pout.emplace(i->name(), pout);
    }
  }

  // Fill in the outputs
  for (auto i : _output_models)
  {
    if (cached_pout.count(i->name()) == 0)
      throw std::runtime_error("Internal error, incorrect dependency resolution");

    out.fill(cached_pout.at(i->name()));

    // Optionally compute the derivatives
    if (dout_din)
      chain_rule(*i, cached_dpout_dpin, *dout_din);
  }
}

void
ComposedModel::chain_rule(const Model & i,
                          const std::map<std::string, LabeledMatrix> & cached_dpout_dpin,
                          LabeledMatrix dout_din) const
{
  const auto & deps = dependent_models(i.name());

  // Base case: If I am a leaf model, the total derivative is just my partial derivative
  if (deps.empty())
  {
    dout_din.fill(cached_dpout_dpin.at(i.name()));
    return;
  }

  TorchSize nbatch = dout_din.batch_size();

  // The partial derivatives of output() w.r.t. input()
  LabeledMatrix dpout_dpin = cached_dpout_dpin.at(i.name());

  // The partial derivatives of the registered model's inputs w.r.t. the total inputs
  LabeledMatrix dpin_din(nbatch, i.input(), input());
  for (auto dep : deps)
    chain_rule(*dep, cached_dpout_dpin, dpin_din);

  // Chain rule
  dout_din.fill(dpout_dpin.chain(dpin_din));
}

void
ComposedModel::resolve_dependency()
{
  // First figure out the leaf models
  _input_models.clear();
  for (const auto & [name, deps] : _dependecies)
    if (deps.empty())
      _input_models.push_back(_models[name]);

  // The leaf models define the inputs
  input().clear();
  for (auto i : _input_models)
    input().merge(i->input());

  // Find the root model(s)
  // Basic idea: if a model is not needed by any other model, then it must be a root model
  _output_models.clear();
  std::map<std::string, bool> visited;
  for (const auto & [name, deps] : _dependecies)
    for (auto dep : deps)
      visited[dep->name()] = true;
  for (const auto & [name, i] : _models)
    if (!visited[name])
      _output_models.push_back(i);

  // The root models define the outputs
  output().clear();
  for (auto i : _output_models)
    output().merge(i->output());

  // Figure out the evaluation order
  _evaluation_order.clear();
  visited.clear();
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
