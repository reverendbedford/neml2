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

#include "neml2/models/ChainRule.h"

namespace neml2
{
ChainRule::ChainRule(const ComposedModel * model)
  : _model(model)
{
}

template <AssemblyMode M>
ChainRuleImpl<M>::ChainRuleImpl(const ComposedModel * model)
  : ChainRule(model)
{
}

template <AssemblyMode M>
void
ChainRuleImpl<M>::clear()
{
  _dout_din.clear();
  _d2out_din2.clear();
}

template <AssemblyMode M>
void
ChainRuleImpl<M>::allocate_variables()
{
  auto I = BatchTensor::identity(
      _model->batch_sizes(), _model->input_axis().storage_size(), _model->options());
  _din_din = Deriv_t(I, {&_model->input_axis(), &_model->input_axis()});
}

template <AssemblyMode M>
void
ChainRuleImpl<M>::apply(bool second_order)
{
  for (auto i : _model->registered_models())
  {
    Deriv_t dpin_din(
        _model->batch_sizes(), {&i->input_axis(), &_model->input_axis()}, _model->options());
    dpin_din.collect_(_din_din);
    if (_model->dependency().node_providers().count(i))
      for (auto dep : _model->dependency().node_providers().at(i))
        dpin_din.collect_(_dout_din[dep]);
    _dout_din[i] = Deriv_t::chain(i->derivative_storage<Deriv_t>(), dpin_din);

    if (second_order)
    {
      SecDeriv_t d2pin_din2(_model->batch_sizes(),
                            {&i->input_axis(), &_model->input_axis(), &_model->input_axis()},
                            _model->options());
      if (_model->dependency().node_providers().count(i))
        for (auto dep : _model->dependency().node_providers().at(i))
          d2pin_din2.collect_(_d2out_din2[dep]);
      _d2out_din2[i] = SecDeriv_t::chain(i->second_derivative_storage<SecDeriv_t>(),
                                         d2pin_din2,
                                         i->derivative_storage<Deriv_t>(),
                                         dpin_din);
    }
  }
}

template <AssemblyMode M>
const StorageTensor<2> &
ChainRuleImpl<M>::total_derivative(Model * i) const
{
  return _dout_din.at(i);
}

template <AssemblyMode M>
const StorageTensor<3> &
ChainRuleImpl<M>::total_second_derivative(Model * i) const
{
  return _d2out_din2.at(i);
}

template class ChainRuleImpl<AssemblyMode::INPLACE>;
}
