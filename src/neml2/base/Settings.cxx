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

#include "neml2/base/Settings.h"

namespace neml2
{
OptionSet
Settings::expected_options()
{
  OptionSet options;
  options.section() = "Settings";
  options.doc() = "Global settings for tensors, models, etc.";

  options.set<std::string>("type") = "Settings";

  EnumSelection dtype_selection({"Float16", "Float32", "Float64"},
                                {static_cast<int>(torch::kFloat16),
                                 static_cast<int>(torch::kFloat32),
                                 static_cast<int>(torch::kFloat64)},
                                "Float64");
  options.set<EnumSelection>("default_floating_point_type") = dtype_selection;
  options.set("default_floating_point_type").doc() =
      "Default floating point type for tensors. Options are " + dtype_selection.candidates_str();

  EnumSelection int_dtype_selection({"Int8", "Int16", "Int32", "Int64"},
                                    {static_cast<int>(torch::kInt8),
                                     static_cast<int>(torch::kInt16),
                                     static_cast<int>(torch::kInt32),
                                     static_cast<int>(torch::kInt64)},
                                    "Int64");
  options.set<EnumSelection>("default_integer_type") = int_dtype_selection;
  options.set("default_integer_type").doc() =
      "Default integer type for tensors. Options are " + int_dtype_selection.candidates_str();

  options.set<std::string>("default_device") = "cpu";
  options.set("default_device").doc() =
      "Default device on which tensors are created and models are evaluated. The string supplied "
      "must follow the following schema: (cpu|cuda)[:<device-index>] where cpu or cuda specifies "
      "the device type, and :<device-index> optionally specifies a device index. For example, "
      "device='cpu' sets the target compute device to be CPU, and device='cuda:1' sets the target "
      "compute device to be CUDA with device ID 1.";

  return options;
}

Settings::Settings(const OptionSet & options)
{
  // Default floating point dtype
  default_dtype() = options.get<EnumSelection>("default_floating_point_type").as<torch::Dtype>();
  torch::set_default_dtype(scalarTypeToTypeMeta(default_dtype()));

  // Default integral dtype
  default_int_dtype() = options.get<EnumSelection>("default_integer_type").as<torch::Dtype>();

  // Default device
  default_device() = torch::Device(options.get<std::string>("default_device"));
}
} // namespace neml2
