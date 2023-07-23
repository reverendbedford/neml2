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

#include "neml2/base/ExtractParametersWalker.h"
#include "neml2/misc/parser_utils.h"

namespace neml2
{
void
ExtractParamsWalker::walk(const std::string & fullpath, const std::string & nodepath, hit::Node * n)
{
#define extract_param_base(ptype, method)                                                          \
  else if (param->type() ==                                                                        \
           utils::demangle(                                                                        \
               typeid(ptype).name())) dynamic_cast<ParameterSet::Parameter<ptype> *>(param.get())  \
      ->set() = method(n->strVal())

#define extract_param_t(ptype)                                                                     \
  extract_param_base(ptype, utils::parse<ptype>);                                                  \
  extract_param_base(std::vector<ptype>, utils::parse_vector<ptype>);                              \
  extract_param_base(std::vector<std::vector<ptype>>, utils::parse_vector_vector<ptype>)

  if (n->type() == hit::NodeType::Field)
  {
    bool found = false;
    for (auto & [name, param] : _params)
      if (name == nodepath)
      {
        found = true;

        if (false)
          ;
        extract_param_t(bool);
        extract_param_t(int);
        extract_param_t(unsigned int);
        extract_param_t(TorchSize);
        extract_param_t(Real);
        extract_param_t(std::string);
        extract_param_t(LabeledAxisAccessor);
        extract_param_t(CrossRef<torch::Tensor>);
        extract_param_t(CrossRef<Scalar>);
        extract_param_t(CrossRef<SymR2>);
        else neml_assert(false, "Unsupported parameter type for parameter ", fullpath);

        break;
      }
    neml_assert(found, "Unused parameter ", fullpath);
  }
}
} // namespace neml2
