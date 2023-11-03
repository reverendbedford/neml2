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

#include "neml2/tensors/BatchTensor.h"
#include "neml2/misc/parser_utils.h"

namespace neml2
{
/**
 * @brief The base class to allow us set up a containers of BatchTensors while maintaining
 * polymorphism. The concrete definition below `BatchTensorValue` will be templated on the actual
 * tensor type.
 *
 */
class BatchTensorValueBase
{
public:
  virtual ~BatchTensorValueBase() = default;

  /**
   * String identifying the type of parameter stored.
   * Must be reimplemented in derived classes.
   */
  virtual std::string type() const = 0;

  /// Send the value to the target device
  virtual void to(const torch::Device &) = 0;

  /// Convert the parameter value to a BatchTensor
  virtual operator BatchTensor() const = 0;
};

/// Concrete definition of a BatchTensor value
template <typename T>
class BatchTensorValue : public BatchTensorValueBase
{
public:
  BatchTensorValue() = default;

  BatchTensorValue(const T & value)
    : _value(value)
  {
  }

  virtual std::string type() const override { return utils::demangle(typeid(T).name()); }

  virtual void to(const torch::Device & device) override { _value = _value.to(device); }

  virtual operator BatchTensor() const override { return BatchTensor(_value); }

  const T & get() const { return _value; }

  T & set() { return _value; }

private:
  /// Stored BatchTensor
  T _value;
};
} // namespace neml2
