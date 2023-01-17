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

class StructuralDriver
{
public:
  // TODO: Add temperature as an input
  StructuralDriver(const neml2::Model & model,
                   torch::Tensor time,
                   torch::Tensor driving_force,
                   std::string driving_force_name);

  /// Actually run and return the results
  virtual std::tuple<std::vector<neml2::LabeledVector>, std::vector<neml2::LabeledVector>> run();

protected:
  virtual neml2::LabeledVector solve_step(neml2::LabeledVector in,
                                          neml2::TorchSize i,
                                          std::vector<neml2::LabeledVector> & all_inputs,
                                          std::vector<neml2::LabeledVector> & all_outputs) const;

  const neml2::Model & _model;
  torch::Tensor _time;
  torch::Tensor _driving_force;
  std::string _driving_force_name;
  neml2::TorchSize _nsteps;
  neml2::TorchSize _nbatch;
};
