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

#pragma once

#include <torch/jit.h>
#include <torch/csrc/jit/frontend/tracer.h>

#include "neml2/misc/types.h"
#include "neml2/misc/error.h"

namespace neml2
{
/// Assert that we are currently tracing
void neml_assert_tracing();

/// Assert that we are currently NOT tracing
void neml_assert_not_tracing();

/// Assert that we are currently tracing (only effective in debug mode)
void neml_assert_tracing_dbg();

/// Assert that we are currently NOT tracing (only effective in debug mode)
void neml_assert_not_tracing_dbg();

namespace jit
{
/// Convert a tuple into a stack
template <typename... Args>
torch::jit::Stack tuple_to_stack(const std::tuple<Args...> & tuple);

/// Convert a stack into a tuple (the stack is consumed)
template <typename... Args>
std::tuple<Args...> stack_to_tuple(torch::jit::Stack & stack);

namespace details
{
template <size_t i, typename... Args>
void stack_to_tuple_impl(torch::jit::Stack & stack, std::tuple<Args...> & tuple);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename... Args>
torch::jit::Stack
tuple_to_stack(const std::tuple<Args...> & tuple)
{
  torch::jit::Stack stack;
  stack.reserve(sizeof...(Args));
  std::apply([&stack](auto &&... args)
             { (stack.push_back(std::forward<decltype(args)>(args)), ...); },
             tuple);
  return stack;
}

template <typename... Args>
std::tuple<Args...>
stack_to_tuple(torch::jit::Stack & stack)
{
  std::tuple<Args...> tuple;
  details::stack_to_tuple_impl<sizeof...(Args) - 1, Args...>(stack, tuple);
  stack.clear();
  return tuple;
}

namespace details
{
template <size_t i, typename... Args>
void
stack_to_tuple_impl(torch::jit::Stack & stack, std::tuple<Args...> & tuple)
{
  using T = typename std::tuple_element_t<i, std::tuple<Args...>>;
  std::get<i>(tuple) = std::move(stack[i]).template to<T>();
  if constexpr (i > 0)
    stack_to_tuple_impl<i - 1, Args...>(stack, tuple);
}
} // namespace details
} // namespace jit
} // namespace neml2
