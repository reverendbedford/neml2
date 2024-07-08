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

#include "neml2/models/ComposedModel.h"
#include "neml2/tensors/StorageTensorType.h"

namespace neml2
{
class ChainRule
{
public:
  ChainRule(const ComposedModel *);

  virtual ~ChainRule() = default;

  virtual void clear() = 0;

  virtual void allocate_variables() = 0;

  virtual void apply(bool) = 0;

  virtual const StorageTensor<2> & total_derivative(Model *) const = 0;

  virtual const StorageTensor<3> & total_second_derivative(Model *) const = 0;

protected:
  const ComposedModel * _model;
};

template <AssemblyMode M>
class ChainRuleImpl : public ChainRule
{
public:
  ChainRuleImpl(const ComposedModel *);

  virtual void clear() override;

  virtual void allocate_variables() override;

  virtual void apply(bool) override;

  virtual const StorageTensor<2> & total_derivative(Model *) const override;

  virtual const StorageTensor<3> & total_second_derivative(Model *) const override;

private:
  using Deriv_t = typename DerivativeStorageType<M>::type;
  using SecDeriv_t = typename SecondDerivativeStorageType<M>::type;

  Deriv_t _din_din;
  std::map<Model *, Deriv_t> _dout_din;
  std::map<Model *, SecDeriv_t> _d2out_din2;
};
}
