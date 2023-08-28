import yaml
import sys
import re


def get_type(params):
    for param in params:
        for param_name, info in param.items():
            if param_name == "type":
                return demangle(info["value"])


def demangle(type):
    type = type.replace(
        "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >",
        "std::string",
    )
    type = re.sub(", std::allocator<.+> ", "", type)
    type = type.replace("neml2::", "")
    type = type.replace("std::", "")
    type = type.replace("at::", "")
    type = re.sub("CrossRef<(.+)>", r"\1", type)
    return type


def postprocess(value, type):
    if type == "bool":
        value = str(bool(value))
    return value


with open("syntax.yml", "r") as stream:
    syntax = yaml.safe_load(stream)

with open(sys.argv[1], "w") as stream:
    stream.write("# Syntax Documentation {#syntax}\n\n")
    stream.write("[TOC]\n\n")
    for s in syntax:
        for type, params in s.items():
            input_type = get_type(params)
            stream.write("## {}\n\n".format(input_type))
            names = []
            types = []
            values = []
            for param in params:
                for param_name, info in param.items():
                    if param_name == "name":
                        continue
                    if param_name == "type":
                        continue
                    param_type = demangle(info["type"])
                    param_value = postprocess(info["value"], param_type)
                    stream.write("- {}\n".format(param_name))
                    stream.write("  - **Type**: {}\n".format(param_type))
                    if param_value != None:
                        stream.write("  - **Default**: {}\n".format(param_value))
            stream.write("\n")
            stream.write("Details: [{}](@ref {})\n\n".format(input_type, type))
