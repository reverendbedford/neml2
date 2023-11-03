#include "neml2/models/ParameterStore.h"
#include "neml2/models/NonlinearParameter.h"

namespace neml2
{

template <typename T, typename>
const T &
ParameterStore::declare_parameter(const std::string & name, const std::string & input_option_name)
{
  if (_options.contains<T>(input_option_name))
    return declare_parameter(name, _options.get<T>(input_option_name));
  else if (_options.contains<CrossRef<T>>(input_option_name))
  {
    try
    {
      return declare_parameter(name, T(_options.get<CrossRef<T>>(input_option_name)));
    }
    catch (const NEMLException & e1)
    {
      try
      {
        // Handle the case of *nonlinear* parameter.
        // Note that nonlinear parameter should only exist inside a Model.
        auto model = dynamic_cast<Model *>(this);
        if (model)
        {
          auto & nl_param = Factory::get_object<NonlinearParameter<T>>(
              "Models", _options.get<CrossRef<T>>(input_option_name).raw());
          model->declare_input_variable<T>(nl_param.p);
          _nl_params.emplace(name, nl_param.p);
          return nl_param.get_value();
        }
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

template const Scalar & ParameterStore::declare_parameter<Scalar>(const std::string &,
                                                                  const std::string &);
template const Vec & ParameterStore::declare_parameter<Vec>(const std::string &,
                                                            const std::string &);
template const Rot & ParameterStore::declare_parameter<Rot>(const std::string &,
                                                            const std::string &);
template const R2 & ParameterStore::declare_parameter<R2>(const std::string &, const std::string &);
template const SR2 & ParameterStore::declare_parameter<SR2>(const std::string &,
                                                            const std::string &);
template const R3 & ParameterStore::declare_parameter<R3>(const std::string &, const std::string &);
template const SFR3 & ParameterStore::declare_parameter<SFR3>(const std::string &,
                                                              const std::string &);
template const R4 & ParameterStore::declare_parameter<R4>(const std::string &, const std::string &);
template const SSR4 & ParameterStore::declare_parameter<SSR4>(const std::string &,
                                                              const std::string &);
template const R5 & ParameterStore::declare_parameter<R5>(const std::string &, const std::string &);
template const SSFR5 & ParameterStore::declare_parameter<SSFR5>(const std::string &,
                                                                const std::string &);
} // namespace neml2
