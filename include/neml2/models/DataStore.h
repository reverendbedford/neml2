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

#include "neml2/base/NEML2Object.h"

#include "neml2/base/Factory.h"
#include "neml2/models/ContainsBuffers.h"

namespace neml2
{

/// @brief  Parent class for all models that store data and can be sent with .to
class DataStore : public NEML2Object
{
public:
  static OptionSet expected_options();

  /// Setup from options
  DataStore(const OptionSet & options);

  /**
   * @brief Recursively send this model and its sub-models to the target device.
   *
   * @param device The target device
   */
  virtual void to(const torch::Device & device);

  /// By default we have no parameters
  virtual std::map<std::string, BatchTensor> named_parameters(bool recurse = false) const
  {
    (void)recurse;
    return {};
  };

  /// The models that may be used during the evaluation of this model
  const std::vector<DataStore *> & registered_data_stores() const
  {
    return _registered_data_stores;
  }

protected:
  /**
  Register a model that the current model may use during its evaluation. No dependency information
  is added.

  NOTE: We also register this model as a submodule (in torch's language), so that when *this*
  `Model` is sent to another device, the registered `Model` is also sent to that device.
  */
  void register_data_store(std::shared_ptr<DataStore> model);

  /**
   * Both register a model and return a reference
   */
  template <typename T, typename = typename std::enable_if_t<std::is_base_of_v<DataStore, T>>>
  T & register_data_store(const std::string & name)
  {
    std::shared_ptr<DataStore> model = Factory::get_object_ptr<DataStore>("Data", name);

    register_data_store(model);

    return *(std::dynamic_pointer_cast<T>(model));
  }

  /// Models *this* model may use during its evaluation
  std::vector<DataStore *> _registered_data_stores;
};

typedef ContainsBuffers<DataStore> Data;

} // namespace neml2