#include "neml2/base/HITParser.h"

namespace neml2
{
void
HITParser::parse(const std::string & filename)
{
  std::ifstream file(filename);
  neml_assert(file.is_open(), "Unable to open file ", filename);

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string input = buffer.str();

  _root.reset(dynamic_cast<hit::Section *>(hit::parse("Hit parser", input)));

  extract_params();
}

void
HITParser::extract_params()
{
  for (const auto & section : Factory::pipeline)
  {
    auto section_node = _root->find(section);
    if (section_node)
    {
      auto objects = section_node->children(hit::NodeType::Section);
      for (auto object : objects)
      {
        // The object name is its node path
        std::string name = object->path();

        // There is a special field reserved for object type
        std::string type = object->param<std::string>("type");

        // Retrieve the expected parameters for this type
        ParameterSet params = Registry::expected_params(type);
        params.set<std::string>("name") = name;
        params.set<std::string>("type") = type;

        // Extract other parameters
        ExtractParamsWalker epw(params);
        object->walk(&epw);

        _all_params[section][name] = params;
      }
    }
  }
}

void
HITParser::ExtractParamsWalker::walk(const std::string & fullpath,
                                     const std::string & nodepath,
                                     hit::Node * n)
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
        extract_param_t(Real);
        extract_param_t(std::string);

        break;
      }
    neml_assert(found, "Unused parameter ", fullpath);
  }
}
} // namespace neml2
