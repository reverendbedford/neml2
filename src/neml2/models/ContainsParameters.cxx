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

#include "neml2/models/ContainsParameters.h"
#include "neml2/base/Factory.h"
#include "neml2/misc/parser_utils.h"
#include "neml2/models/ContainsBuffers.h"
#include "neml2/models/Model.h"

#include "neml2/tensors/tensors.h"
#include "neml2/models/NonlinearParameter.h"

namespace neml2
{

template <typename Base>
void
ContainsParameters<Base>::to(const torch::Device & device)
{
  Base::to(device);

  for (auto && [name, id] : _param_ids)
    _param_values[id].to(device);
}

template <typename Base>
std::map<std::string, BatchTensor>
ContainsParameters<Base>::named_parameters(bool recurse) const
{
  std::map<std::string, BatchTensor> params;

  for (const auto & n : _param_names)
    params.emplace(n, _param_values[_param_ids.at(n)]);

  if (recurse)
    for (auto & model : Base::registered_data_stores())
      for (auto && [n, v] : model->named_parameters(true))
        params.emplace(model->name() + "." + n, v);

  return params;
}

template <typename Base>
template <typename T, typename>
const T &
ContainsParameters<Base>::declare_parameter(const std::string & name,
                                            const std::string & input_option_name)
{
  if (Base::options().template contains<T>(input_option_name))
    return declare_parameter(name, Base::options().template get<T>(input_option_name));
  else if (Base::options().template contains<CrossRef<T>>(input_option_name))
  {
    try
    {
      return declare_parameter(name,
                               T(Base::options().template get<CrossRef<T>>(input_option_name)));
    }
    catch (const NEMLException & e1)
    {
      try
      {
        auto & nl_param = Factory::get_object<NonlinearParameter<T>>(
            "Models", Base::options().template get<CrossRef<T>>(input_option_name).raw());
        // This assumes too much
        Base::template declare_input_variable<T>(nl_param.p);
        _nl_params.emplace(name, nl_param.p);
        return nl_param.get_value();
      }
      catch (const NEMLException & e2)
      {
        std::cerr << e1.what() << std::endl;
        std::cerr << e2.what() << std::endl;
      }
    }
  }

  throw NEMLException(
      "Trying to register parameter named " + name + " from input option named " +
      input_option_name + " of type " + utils::demangle(typeid(T).name()) +
      ". Make sure you provided the correct parameter name, option name, and parameter type. Note "
      "that the parameter type can either be a plain type, a cross-reference, or an "
      "interpolation.");
}

template class ContainsParameters<ContainsBuffers<ModelBase>>;

template const Scalar &
ContainsParameters<ContainsBuffers<ModelBase>>::declare_parameter<Scalar>(const std::string &,
                                                                          const std::string &);
template const Vec &
ContainsParameters<ContainsBuffers<ModelBase>>::declare_parameter<Vec>(const std::string &,
                                                                       const std::string &);
template const Rot &
ContainsParameters<ContainsBuffers<ModelBase>>::declare_parameter<Rot>(const std::string &,
                                                                       const std::string &);
template const R2 &
ContainsParameters<ContainsBuffers<ModelBase>>::declare_parameter<R2>(const std::string &,
                                                                      const std::string &);
template const SR2 &
ContainsParameters<ContainsBuffers<ModelBase>>::declare_parameter<SR2>(const std::string &,
                                                                       const std::string &);
template const R3 &
ContainsParameters<ContainsBuffers<ModelBase>>::declare_parameter<R3>(const std::string &,
                                                                      const std::string &);
template const SFR3 &
ContainsParameters<ContainsBuffers<ModelBase>>::declare_parameter<SFR3>(const std::string &,
                                                                        const std::string &);
template const R4 &
ContainsParameters<ContainsBuffers<ModelBase>>::declare_parameter<R4>(const std::string &,
                                                                      const std::string &);
template const SSR4 &
ContainsParameters<ContainsBuffers<ModelBase>>::declare_parameter<SSR4>(const std::string &,
                                                                        const std::string &);
template const R5 &
ContainsParameters<ContainsBuffers<ModelBase>>::declare_parameter<R5>(const std::string &,
                                                                      const std::string &);
template const SSFR5 &
ContainsParameters<ContainsBuffers<ModelBase>>::declare_parameter<SSFR5>(const std::string &,
                                                                         const std::string &);

} // namespace neml2