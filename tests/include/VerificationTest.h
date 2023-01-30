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

#include <string>

#include <torch/torch.h>

#include "neml2/models/Model.h"

class VerificationTest
{
public:
  VerificationTest(std::string fname);

  /// Evaluate the comparison between the two models
  bool compare(const neml2::Model & model) const;

  /// Driving time data
  torch::Tensor time() const { return _time; };

  /// Driving strain data
  torch::Tensor strain() const { return _strain; };

  /// Driving temperature data
  torch::Tensor temperature() const { return _temperature; };

  /// Stress data, for comparison
  torch::Tensor stress() const { return _stress; };

  /// Does this test have actual temperature data?
  bool with_temperature() const { return _with_temperature; };

  /// Relative tolerance
  double rtol() const { return _rtol; };

  /// Absolute tolerance
  double atol() const { return _atol; };

private:
  /// Read the test data
  void parse();

private:
  const std::string _filename;
  std::string _neml_model_file;
  std::string _neml_model_name;
  std::string _neml2_model_file;
  std::string _neml2_model_name;
  double _rtol, _atol;
  std::string _description;
  bool _with_temperature;
  torch::Tensor _time;
  torch::Tensor _strain;
  torch::Tensor _temperature;
  torch::Tensor _stress;
};

std::vector<std::string> split_string(const std::string & input, const char * delimiter = " ");
