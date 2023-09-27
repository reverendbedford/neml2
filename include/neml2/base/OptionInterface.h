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

#include <torch/torch.h>
#include "neml2/base/OptionSet.h"
#include "neml2/base/CrossRef.h"

namespace neml2
{
/**
 * @brief A convenient interface for NEML2Object to work with input options.
 *
 */
class OptionInterface
{
public:
  OptionInterface(const OptionSet & options, torch::nn::Module * object);

  /// Get the options used to generate *this* object.
  const OptionSet & input_options() const { return _options; }

protected:
  /**
   * @brief Register a model parameter.
   *
   * @tparam T Parameter type. See @ref primitive for supported types.
   * @param name Name of the model parameter.
   * @param input_option_name Name of the input option that defines the value of the model
   * parameter.
   * @return T The value of the registered model parameter.
   */
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  T register_model_parameter(const std::string & name, const std::string & input_option_name);

  /**
   * @brief Register a cross-referenced model parameter.
   *
   * @tparam T Parameter type. See @ref primitive for supported types.
   * @param name Name of the model parameter.
   * @param input_option_name Name of the input option that defines the value of the model
   * parameter.
   * @return T The value of the registered model parameter.
   */
  template <typename T,
            typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
  T register_crossref_model_parameter(const std::string & name,
                                      const std::string & input_option_name);

private:
  const OptionSet & _options;

  torch::nn::Module * _object;
};

template <typename T,
          typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
T
OptionInterface::register_model_parameter(const std::string & name,
                                          const std::string & input_option_name)
{
  auto p = _options.get<T>(input_option_name);
  return _object->register_parameter(name, p, /*requires_grad=*/false);
}

template <typename T,
          typename = typename std::enable_if_t<std::is_base_of_v<BatchTensorBase<T>, T>>>
T
OptionInterface::register_crossref_model_parameter(const std::string & name,
                                                   const std::string & input_option_name)
{
  auto p = T(_options.get<CrossRef<T>>(input_option_name));
  return T(_object->register_parameter(name, p, /*requires_grad=*/false), p.batch_dim());
}
} // namespace neml2
