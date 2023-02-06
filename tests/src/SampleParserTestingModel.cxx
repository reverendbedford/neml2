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

#include "SampleParserTestingModel.h"

using namespace neml2;

register_NEML2_object(SampleParserTestingModel);

ParameterSet
SampleParserTestingModel::expected_params()
{
  ParameterSet params = Model::expected_params();

  // Integral types we support:
  //   bool
  //   int
  //   unsigned int
  //   Real
  //   string
  //
  // We also support vector of and vector of vector of each integral type.
  params.set<bool>("bool");
  params.set<std::vector<bool>>("bool_vec");
  params.set<std::vector<std::vector<bool>>>("bool_vec_vec");

  params.set<int>("int");
  params.set<std::vector<int>>("int_vec");
  params.set<std::vector<std::vector<int>>>("int_vec_vec");

  params.set<unsigned int>("uint");
  params.set<std::vector<unsigned int>>("uint_vec");
  params.set<std::vector<std::vector<unsigned int>>>("uint_vec_vec");

  params.set<Real>("Real");
  params.set<std::vector<Real>>("Real_vec");
  params.set<std::vector<std::vector<Real>>>("Real_vec_vec");

  params.set<std::string>("string");
  params.set<std::vector<std::string>>("string_vec");
  params.set<std::vector<std::vector<std::string>>>("string_vec_vec");

  return params;
}
