#pragma once

#include "models/Model.h"

/// A model composed of other models. Users will have to specify the dependencies among all the
/// models. The dependencies will be maintained and sorted as a directed-acyclic graph (DAG).
class ComposedModel : public Model
{
public:
  ComposedModel(
      const std::string & name,
      const std::vector<std::pair<std::shared_ptr<Model>, std::shared_ptr<Model>>> & dependencies);

  /// Return dependencies of a registered model
  const std::vector<std::shared_ptr<Model>> & dependent_models(const std::string & name) const;

  /// Stringify the evaluation order
  std::string evaluation_order() const;

  /// Write the entire model graph in dot format
  void to_dot(std::ostream & os) const;

protected:
  /// Evaluate the model graph all the way up from the leaf models
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  /// Add a node in the DAG
  void add_node(const std::shared_ptr<Model> & model);

  /// Register a dependency, e.g., adding a directed edge in the DAG
  void register_dependency(const std::shared_ptr<Model> & from, const std::shared_ptr<Model> & to);

  /// Resolve dependency using topological traversal of the dependent models
  void resolve_dependency();

  /// Nodes of this DAG
  std::unordered_map<std::string, std::shared_ptr<Model>> _models;

  /// Dependencies among the nodes, e.g., the directed edges of the DAG
  /// This is also referred to as the adjacency matrix of a graph
  std::unordered_map<std::string, std::vector<std::shared_ptr<Model>>> _dependecies;

  /// Leaf nodes in the DAG -- they define the inputs of this composed model graph
  std::vector<std::shared_ptr<Model>> _input_models;

  /// Root node(s) in the DAG -- they define the outputs of this composed model graph
  std::vector<std::shared_ptr<Model>> _output_models;

  /// The order which we can follow to evaluate all the registered models
  std::vector<std::shared_ptr<Model>> _evaluation_order;

private:
  /// Helper function to recurse the model graph to evaluate the total derivative
  void chain_rule(const std::shared_ptr<Model> & i,
                  const std::unordered_map<std::string, LabeledMatrix> & cached_dpout_dpin,
                  LabeledMatrix dout_din) const;

  /// Dependency resolution helper
  void resolve_dependency(const std::shared_ptr<Model> & i,
                          std::vector<std::shared_ptr<Model>> & dep,
                          std::unordered_map<std::string, bool> & visited);

  /// Helper function to write this model only
  void to_dot(std::ostream & os,
              const std::shared_ptr<Model> & model,
              int & id,
              std::unordered_map<std::string, int> & io_ids) const;
};
