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

#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/jit.h>

namespace neml2
{
/**
 * @brief A traced function with static call signature (input and output types determined at
 * compile-time).
 *
 * @tparam OutputTypes Tuple of output types
 * @tparam InputTypes Tuple of input types
 */
template <typename OutputTypes, typename InputTypes>
class StaticGraphFunction
{
public:
  /**
   * @brief Construct a new StaticGraphFunction object by tracing the given function with the given
   * example inputs
   *
   * @param name Function name
   * @param f Function to trace
   * @param x Example inputs
   */
  StaticGraphFunction(c10::QualifiedName name,
                      const std::function<OutputTypes(InputTypes)> & f,
                      const InputTypes & x);

  /// Call the traced function and return the outputs
  template <typename... Args>
  OutputTypes operator()(Args &&... args);

private:
  /// Helper to create the traced function graph
  std::shared_ptr<torch::jit::Graph> create_graph(const std::function<OutputTypes(InputTypes)> & f,
                                                  const InputTypes & x) const;

  /// Helper to create the stack for the tracer
  torch::jit::Stack make_stack(const InputTypes & x) const;

  /// Helper to recursively extract outputs from the stack
  template <size_t i>
  void extract_outputs(torch::jit::Stack & stack, OutputTypes & y) const
  {
    using T = typename std::tuple_element<i, OutputTypes>::type;
    std::get<i>(y) = std::move(stack[i]).template to<T>();
    if constexpr (i > 0)
      extract_outputs<i - 1>(stack, y);
  }

  /// The function responsible for handling the function call
  torch::jit::GraphFunction _graph_function;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename OutputTypes, typename InputTypes>
StaticGraphFunction<OutputTypes, InputTypes>::StaticGraphFunction(
    c10::QualifiedName name, const std::function<OutputTypes(InputTypes)> & f, const InputTypes & x)
  : _graph_function(name, std::move(create_graph()), nullptr)
{
}

template <typename OutputTypes, typename InputTypes>
template <typename... Args>
OutputTypes
StaticGraphFunction<OutputTypes, InputTypes>::operator()(Args &&... args)
{
  auto x = std::make_tuple(std::forward<Args>(args)...);
  auto stack = make_stack(x);
  _graph_function.run(stack);
  OutputTypes y;
  neml_assert_dbg(stack.size() == std::tuple_size_v<OutputTypes>,
                  "Number of outputs on the stack does not match the expected number of outputs");
  extract_outputs<std::tuple_size_v<OutputTypes> - 1>(stack, y);
  torch::jit::drop(stack, stack.size());
  return y;
}

template <typename OutputTypes, typename InputTypes>
std::shared_ptr<torch::jit::Graph>
StaticGraphFunction<OutputTypes, InputTypes>::create_graph(
    const std::function<OutputTypes(InputTypes)> & f, const InputTypes & x) const
{
  auto var_name_lookup_fn = [](const torch::autograd::Variable & /*var*/) -> std::string
  { return ""; };
  return std::move(std::get<0>(torch::jit::tracer::trace(make_stack(x),
                                                         f,
                                                         var_name_lookup_fn,
                                                         /*strict=*/true,
                                                         /*force_outplace*/ false))
                       ->graph);
}

template <typename OutputTypes, typename InputTypes>
torch::jit::Stack
StaticGraphFunction<OutputTypes, InputTypes>::make_stack(const InputTypes & x) const
{
  torch::jit::Stack stack;
  stack.reserve(std::tuple_size_v<InputTypes>);
  std::apply(
      [&stack](auto &&... args) { (stack.push_back(std::forward<decltype(args)>(args)), ...); }, x);
  return stack;
}

} // namespace neml2
