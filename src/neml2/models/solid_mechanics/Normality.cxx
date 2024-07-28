// Copyright 2024, UChicago Argonne, LLC
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

#include "neml2/models/solid_mechanics/Normality.h"
#include "neml2/base/guards.h"

namespace neml2
{
register_NEML2_object(Normality);

OptionSet
Normality::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Store the first derivatives of a scalar-valued function in given variables, "
                  "i.e. \\f$ u_i = \\dfrac{f(\\boldsymbol{v})}{v_i} \\f$.";

  options.set<std::string>("model");
  options.set("model").doc() = "The model which evaluates the scalar-valued function";

  options.set<VariableName>("function");
  options.set("function").doc() = "Function to take derivative";

  options.set<std::vector<VariableName>>("from");
  options.set("from").doc() = "Function arguments to take derivatives w.r.t.";

  options.set<std::vector<VariableName>>("to");
  options.set("to").doc() = "Variables to store the first derivatives";

  return options;
}

Normality::Normality(const OptionSet & options)
  : Model(options),
    _model(register_model<Model>(options.get<std::string>("model"), /*extra_deriv_order=*/1)),
    _f(options.get<VariableName>("function"))
{
  // Set up the conjugate pairs
  const auto from = options.get<std::vector<VariableName>>("from");
  const auto to = options.get<std::vector<VariableName>>("to");
  neml_assert(from.size() == to.size(),
              "The conjugate pairs should have a one-to-one correspondance. ",
              from.size(),
              " variables are being mapped to ",
              to.size(),
              " variables.");
  for (size_t i = 0; i < from.size(); i++)
  {
    auto sz = _model.output_axis().storage_size(_f) * _model.input_axis().storage_size(from[i]);
    _conjugate_pairs.emplace(from[i],
                             &declare_output_variable(sz, input_variable(from[i])->type(), to[i]));
  }
}

void
Normality::setup_output_views()
{
  Model::setup_output_views();

  for (auto && [ivar, var] : _conjugate_pairs)
  {
    _f_deriv_views[ivar] = _model.derivative_storage().base_index({_f, ivar});
    if (requires_grad())
      for (auto && [jvar, j] : input_variables())
        _f_secderiv_views[ivar][jvar] =
            _model.second_derivative_storage().base_index({_f, ivar, jvar});
  }
}

void
Normality::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Normality doesn't implement second derivatives.");

  {
    SolvingNonlinearSystem not_solving(false);
    if (out && !dout_din)
      _model.dvalue();
    else
      _model.dvalue_and_d2value();
  }

  for (auto && [ivar, var] : _conjugate_pairs)
  {
    if (out)
      (*var) = _f_deriv_views[ivar];

    if (dout_din)
      for (auto && [jvar, j] : input_variables())
        if (j.is_dependent())
          var->d(j) = _f_secderiv_views[ivar][jvar];
  }
}
} // namespace neml2
