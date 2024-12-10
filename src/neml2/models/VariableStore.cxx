// Copyright 2024, UChicago Argonne, LLC
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

#include "neml2/models/VariableStore.h"
#include "neml2/models/Model.h"

namespace neml2
{
VariableStore::VariableStore(const OptionSet & options, Model * object)
  : _object(object),
    _object_options(options),
    _input_axis(declare_axis("input")),
    _output_axis(declare_axis("output")),
    _tensor_options(default_tensor_options())
{
}

LabeledAxis &
VariableStore::declare_axis(const std::string & name)
{
  neml_assert(!_axes.has_key(name),
              "Trying to declare an axis named ",
              name,
              ", but an axis with the same name already exists.");

  auto axis = std::make_unique<LabeledAxis>();
  return *_axes.set_pointer(name, std::move(axis));
}

void
VariableStore::setup_layout()
{
  input_axis().setup_layout();
  output_axis().setup_layout();
}

VariableBase &
VariableStore::input_variable(const VariableName & name)
{
  auto * var_ptr = _input_variables.query_value(name);
  neml_assert(var_ptr, "Input variable ", name, " does not exist in model ", _object->name());
  return *var_ptr;
}

const VariableBase &
VariableStore::input_variable(const VariableName & name) const
{
  const auto * var_ptr = _input_variables.query_value(name);
  neml_assert(var_ptr, "INput variable ", name, " does not exist in model ", _object->name());
  return *var_ptr;
}

VariableBase &
VariableStore::output_variable(const VariableName & name)
{
  auto * var_ptr = _output_variables.query_value(name);
  neml_assert(var_ptr, "Output variable ", name, " does not exist in model ", _object->name());
  return *var_ptr;
}

const VariableBase &
VariableStore::output_variable(const VariableName & name) const
{
  const auto * var_ptr = _output_variables.query_value(name);
  neml_assert(var_ptr, "Output variable ", name, " does not exist in model ", _object->name());
  return *var_ptr;
}

void
VariableStore::clear_input()
{
  for (auto && [name, var] : input_variables())
    if (var.owning())
      var.clear();
}

void
VariableStore::clear_output()
{
  for (auto && [name, var] : output_variables())
    if (var.owning())
      var.clear();
}

void
VariableStore::zero_input()
{
  for (auto && [name, var] : input_variables())
    if (var.owning())
      var.zero(_tensor_options);
}

void
VariableStore::zero_output()
{
  for (auto && [name, var] : output_variables())
    if (var.owning())
      var.zero(_tensor_options);
}

void
VariableStore::assign_input(const std::map<VariableName, Tensor> & vals)
{
  for (const auto & [name, val] : vals)
    input_variable(name).set(val.clone());
}

void
VariableStore::assign_output(const std::map<VariableName, Tensor> & vals)
{
  for (const auto & [name, val] : vals)
    output_variable(name).set(val.clone());
}

void
VariableStore::assign_output_derivatives(
    const std::map<VariableName, std::map<VariableName, Tensor>> & derivs)
{
  for (const auto & [yvar, deriv] : derivs)
    output_variable(yvar).derivatives().insert(deriv.begin(), deriv.end());
}

void
VariableStore::assign_input_stack(const torch::jit::Stack & stack)
{
  const auto & vars = input_axis().variable_names();
  neml_assert_dbg(stack.size() == vars.size(),
                  "Number of input variables in the stack (",
                  stack.size(),
                  ") does not match the number of input variables in the model (",
                  vars.size(),
                  ").");
  for (std::size_t i = 0; i < vars.size(); i++)
    input_variable(vars[i]).set(stack[i].toTensor());
}

void
VariableStore::assign_output_stack(const torch::jit::Stack & stack, bool out, bool dout, bool d2out)
{
  neml_assert_dbg(out || dout || d2out,
                  "At least one of the output/derivative flags must be true.");

  neml_assert_dbg(stack.size() == 1,
                  "Output stack must have exactly one element, got ",
                  stack.size(),
                  ". Multiple output tensors are stored in a list.");
  const auto stacklist = stack[0].toTensorVector();

  // With our protocol, the last tensor in the list is the sparsity tensor
  const auto sparsity_tensor = stacklist.back().contiguous();
  neml_assert_dbg(torch::sum(sparsity_tensor).item<Size>() == Size(stacklist.size()) - 1,
                  "Sparsity tensor has incorrect size. Got ",
                  torch::sum(sparsity_tensor).item<Size>(),
                  " expected ",
                  Size(stacklist.size()) - 1);
  const std::vector<std::uint8_t> sparsity(sparsity_tensor.data_ptr<std::uint8_t>(),
                                           sparsity_tensor.data_ptr<std::uint8_t>() +
                                               sparsity_tensor.size(0));

  const auto & yvars = output_axis().variable_names();
  const auto & xvars = input_axis().variable_names();

  // Stack counter
  std::size_t si = 0;

  if (out)
  {
    for (std::size_t i = 0; i < yvars.size(); i++)
    {
      neml_assert_dbg(sparsity[si], "Corrupted sparsity tensor.");
      output_variable(yvars[i]).set(stacklist[si++], /*force=*/true);
    }
  }

  if (dout)
  {
    for (std::size_t i = 0; i < yvars.size(); i++)
    {
      auto & derivs = output_variable(yvars[i]).derivatives();
      for (std::size_t j = 0; j < xvars.size(); j++)
      {
        if (sparsity[si])
        {
          const auto & val = stacklist[si++];
          neml_assert_dbg(val.dim() >= 2,
                          "Derivative tensor d(",
                          yvars[i],
                          ")/d(",
                          xvars[j],
                          ") must have at least 2 dimensions. Got ",
                          val.dim(),
                          ".");
          derivs[xvars[j]] = Tensor(val, val.dim() - 2);
        }
      }
    }
  }

  if (d2out)
  {
    for (std::size_t i = 0; i < yvars.size(); i++)
    {
      auto & derivs = output_variable(yvars[i]).second_derivatives();
      for (std::size_t j = 0; j < xvars.size(); j++)
        for (std::size_t k = 0; k < xvars.size(); k++)
        {
          if (sparsity[si])
          {
            const auto & val = stacklist[si++];
            neml_assert_dbg(val.dim() >= 3,
                            "Second derivative tensor d2(",
                            yvars[i],
                            ")/d(",
                            xvars[j],
                            ")d(",
                            xvars[k],
                            ") must have at least 3 dimensions. Got ",
                            val.dim(),
                            ".");
            derivs[xvars[j]][xvars[k]] = Tensor(val, val.dim() - 3);
          }
        }
    }
  }
}

std::map<VariableName, Tensor>
VariableStore::collect_input() const
{
  std::map<VariableName, Tensor> vals;
  for (auto && [name, var] : input_variables())
    vals[name] = var.tensor();
  return vals;
}

std::map<VariableName, Tensor>
VariableStore::collect_output() const
{
  std::map<VariableName, Tensor> vals;
  for (auto && [name, var] : output_variables())
    vals[name] = var.tensor();
  return vals;
}

std::map<VariableName, std::map<VariableName, Tensor>>
VariableStore::collect_output_derivatives() const
{
  std::map<VariableName, std::map<VariableName, Tensor>> derivs;
  for (auto && [name, var] : output_variables())
    derivs[name] = var.derivatives();
  return derivs;
}

std::map<VariableName, std::map<VariableName, std::map<VariableName, Tensor>>>
VariableStore::collect_output_second_derivatives() const
{
  std::map<VariableName, std::map<VariableName, std::map<VariableName, Tensor>>> sec_derivs;
  for (auto && [name, var] : output_variables())
    sec_derivs[name] = var.second_derivatives();
  return sec_derivs;
}

torch::jit::Stack
VariableStore::collect_input_stack() const
{
  torch::jit::Stack stack;
  const auto & vars = input_axis().variable_names();
  stack.reserve(vars.size());
  for (const auto & name : vars)
    stack.push_back(input_variable(name).tensor());
  return stack;
}

torch::jit::Stack
VariableStore::collect_output_stack(bool out, bool dout, bool d2out) const
{
  neml_assert_dbg(out || dout || d2out,
                  "At least one of the output/derivative flags must be true.");

  const auto & yvars = output_axis().variable_names();
  const auto & xvars = input_axis().variable_names();

  std::vector<torch::Tensor> stacklist;
  std::vector<std::uint8_t> sparsity;

  if (out)
  {
    sparsity.insert(sparsity.end(), yvars.size(), 1);
    for (const auto & yvar : yvars)
      stacklist.push_back(output_variable(yvar).tensor());
  }

  if (dout)
  {
    for (const auto & yvar : yvars)
    {
      const auto & derivs = output_variable(yvar).derivatives();
      for (const auto & xvar : xvars)
      {
        const auto & deriv = derivs.find(xvar);
        sparsity.push_back(deriv != derivs.end() ? 1 : 0);
        if (sparsity.back())
          stacklist.push_back(deriv->second);
      }
    }
  }

  if (d2out)
  {
    for (const auto & yvar : yvars)
    {
      const auto & derivs = output_variable(yvar).second_derivatives();
      for (const auto & x1var : xvars)
      {
        const auto & x1derivs = derivs.find(x1var);
        if (x1derivs != derivs.end())
          for (const auto & x2var : xvars)
          {
            const auto & x1x2deriv = x1derivs->second.find(x2var);
            sparsity.push_back(x1x2deriv != x1derivs->second.end() ? 1 : 0);
            if (sparsity.back())
              stacklist.push_back(x1x2deriv->second);
          }
        else
          sparsity.insert(sparsity.end(), xvars.size(), 0);
      }
    }
  }

  const auto sparsity_tensor = torch::tensor(sparsity, torch::kUInt8);
  neml_assert_dbg(torch::sum(sparsity_tensor).item<Size>() == Size(stacklist.size()),
                  "Sparsity tensor has incorrect size. Got ",
                  torch::sum(sparsity_tensor).item<Size>(),
                  " expected ",
                  Size(stacklist.size()));
  stacklist.push_back(sparsity_tensor);

  return {stacklist};
}

} // namespace neml2
