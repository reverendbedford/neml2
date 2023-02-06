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
#include "neml2/base/Factory.h"

namespace neml2
{
typedef std::vector<std::pair<std::shared_ptr<Model>, std::shared_ptr<Model>>> ModelDependency;

class ComposedModel : public Model
{
public:
  static ParameterSet expected_params();

  ComposedModel(const ParameterSet & params);

  /// Return dependencies of a registered model
  const std::vector<std::shared_ptr<Model>> & dependent_models(const std::string & name) const;

  /// Stringify the evaluation order
  std::string evaluation_order() const;

  /// Write the entire model graph in dot format
  void to_dot(std::ostream & os) const;

  /// Write the adjacency list
  void print_dependency(std::ostream & os) const;

protected:
  /// Evaluate the model graph all the way up from the leaf models
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  void register_dependency(const std::vector<std::shared_ptr<Model>> & models);

  /// Register a dependency, e.g., adding a directed edge in the DAG
  void add_edge(const std::shared_ptr<Model> & from, const std::shared_ptr<Model> & to);

  /// Add a node in the DAG
  void add_node(const std::shared_ptr<Model> & model);

  /// Resolve dependency using topological traversal of the dependent models
  void resolve_dependency();

  /// Nodes of this DAG
  std::map<std::string, std::shared_ptr<Model>> _models;

  /// Dependencies among the nodes, e.g., the directed edges of the DAG
  /// This is also referred to as the adjacency matrix of a graph
  std::map<std::string, std::vector<std::shared_ptr<Model>>> _dependecies;

  /// Root node(s) in the DAG -- they define the outputs of this composed model graph
  std::vector<std::shared_ptr<Model>> _output_models;

  /// The order which we can follow to evaluate all the registered models
  std::vector<std::shared_ptr<Model>> _evaluation_order;

private:
  /// Dependency resolution helper
  void resolve_dependency(const std::shared_ptr<Model> & i,
                          std::vector<std::shared_ptr<Model>> & dep,
                          std::map<std::string, bool> & visited);

  /// Helper function to write this model only
  void to_dot(std::ostream & os,
              const Model & model,
              int & id,
              std::map<std::string, int> & io_ids) const;
};
} // namespace neml2
