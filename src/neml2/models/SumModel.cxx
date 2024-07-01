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

#include "neml2/models/SumModel.h"
#include "neml2/tensors/SSR4.h"

namespace neml2
{
register_NEML2_object(ScalarSumModel);
register_NEML2_object(SR2SumModel);

template <typename T>
OptionSet
SumModel<T>::expected_options()
{
  // This is the only way of getting tensor type in a static method like this...
  // Trim 6 chars to remove 'neml2::'
  auto tensor_type = utils::demangle(typeid(T).name()).substr(7);

  OptionSet options = Model::expected_options();
  options.doc() = "Calculate linear combination of multiple " + tensor_type +
                  " tensors as \\f$ u = c_i v_i \\f$ (Einstein summation assumed), where \\f$ c_i "
                  "\\f$ are the coefficients, and \\f$ v_i \\f$ are the variables to be summed.";

  options.set<std::vector<VariableName>>("from_var");
  options.set("from_var").doc() = tensor_type + " tensors to be summed";

  options.set<VariableName>("to_var");
  options.set("to_var").doc() = "The sum";

  options.set<std::vector<CrossRef<Scalar>>>("coefficients") = {};
  options.set("coefficients").doc() = "Weights associated with each variable";

  return options;
}

template <typename T>
SumModel<T>::SumModel(const OptionSet & options)
  : Model(options),
    _to(declare_output_variable<T>("to_var"))
{
  for (auto fv : options.get<std::vector<VariableName>>("from_var"))
    _from.push_back(&declare_input_variable<T>(fv));

  // The number of coefficients can be 0, 1, or N
  //  - 0: The _coefs vector will be filled with ones
  //  - 1: The _coefs vector will be filled with _coefs[0]
  //  - N: N must be equal to the length of _from
  const auto coefs_in = options.get<std::vector<CrossRef<Scalar>>>("coefficients");
  const auto N = _from.size();
  if (coefs_in.size() == 0)
    _coefs = std::vector<const Scalar *>(
        N, &declare_parameter("c", Scalar(1.0, default_tensor_options())));
  else if (coefs_in.size() == 1)
    _coefs = std::vector<const Scalar *>(N, &declare_parameter("c", Scalar(coefs_in[0])));
  else
  {
    neml_assert(coefs_in.size() == N,
                "Number of coefficients must be 0, 1, or N, where N is the number of 'from_var'.");
    _coefs.resize(N);
    for (size_t i = 0; i < N; i++)
      _coefs[i] = &declare_parameter("c_" + utils::stringify(i), Scalar(coefs_in[i]));
  }
}

template <typename T>
void
SumModel<T>::set_value(bool out, bool dout_din, bool d2out_din2)
{
  const auto N = _from.size();

  if (out)
  {
    auto sum = T::zeros(_to.batch_sizes(), options());
    for (size_t i = 0; i < N; i++)
      sum += (*_coefs[i]) * (*_from[i]);
    _to = sum;
  }

  if (dout_din)
    for (size_t i = 0; i < N; i++)
      _to.d(*_from[i]) = (*_coefs[i]) * T::identity_map(options());

  if (d2out_din2)
  {
    // zero
  }
}

template class SumModel<Scalar>;
template class SumModel<SR2>;
} // namespace neml2
