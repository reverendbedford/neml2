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

#include "neml2/solvers/NonlinearSystem.h"

class TestNonlinearSystem : public neml2::NonlinearSystem
{
public:
  static neml2::OptionSet expected_options();
  TestNonlinearSystem(const neml2::OptionSet & options);

  virtual neml2::BatchTensor exact_solution(const neml2::BatchTensor & x) const = 0;
  virtual neml2::BatchTensor guess(const neml2::BatchTensor & x) const = 0;
};

class PowerTestSystem : public TestNonlinearSystem
{
public:
  static neml2::OptionSet expected_options();
  PowerTestSystem(const neml2::OptionSet & options);

  virtual void assemble(const neml2::BatchTensor & x,
                        neml2::BatchTensor * residual,
                        neml2::BatchTensor * Jacobian = nullptr) const;
  virtual neml2::BatchTensor exact_solution(const neml2::BatchTensor & x) const;
  virtual neml2::BatchTensor guess(const neml2::BatchTensor & x) const;
};

class RosenbrockTestSystem : public TestNonlinearSystem
{
public:
  static neml2::OptionSet expected_options();
  RosenbrockTestSystem(const neml2::OptionSet & options);

  virtual void assemble(const neml2::BatchTensor & x,
                        neml2::BatchTensor * residual,
                        neml2::BatchTensor * Jacobian = nullptr) const;
  virtual neml2::BatchTensor exact_solution(const neml2::BatchTensor & x) const;
  virtual neml2::BatchTensor guess(const neml2::BatchTensor & x) const;
};
