#pragma once

#include "models/Model.h"

/// A model composed of other models. Users will have to specify the dependencies among all the
/// models. The dependencies will be maintained and sorted as a directed-acyclic graph (DAG).
class ComposedModel : public Model
{
public:
  ComposedModel(const std::string & name);

  /// Register a model, e.g., adding a node in the DAG
  void registerModel(Model & model);

  /// Register a dependency, e.g., adding a directed edge in the DAG
  void registerDependency(const std::string & from, const std::string & to);

  /// Return dependencies of a registered model
  const std::vector<Model *> & dependent_models(const std::string & name) const;

  /// Stringify the evaluation order
  std::string evaluation_order() const;

  /// Write the entire model graph in dot format
  void to_dot(std::ostream & os) const;

protected:
  /// Evaluate the model graph all the way up from the leaf models
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  virtual void setup();

  /// Resolve dependency using topological traversal of the dependent models
  void resolve_dependency();

  /// Nodes of this DAG
  std::unordered_map<std::string, Model *> _models;

  /// Dependencies among the nodes, e.g., the directed edges of the DAG
  /// This is also referred to as the adjacency matrix of a graph
  std::unordered_map<std::string, std::vector<Model *>> _dependecies;

  /// Leaf nodes in the DAG -- they define the inputs of this composed model graph
  std::vector<Model *> _input_models;

  /// Root node(s) in the DAG -- they define the outputs of this composed model graph
  std::vector<Model *> _output_models;

  /// The order which we can follow to evaluate all the registered models
  std::vector<Model *> _evaluation_order;

private:
  /// Helper function to recurse the model graph to evaluate the total derivative
  void chain_rule(Model * i,
                  const std::unordered_map<std::string, LabeledMatrix> & cached_dpout_dpin,
                  LabeledMatrix dout_din) const;

  /// Dependency resolution helper
  void resolve_dependency(Model * i,
                          std::vector<Model *> & dep,
                          std::unordered_map<Model *, bool> & visited);

  /// Helper function to write this model only
  void to_dot(std::ostream & os,
              Model * model,
              int & id,
              std::unordered_map<std::string, int> & io_ids) const;
};
