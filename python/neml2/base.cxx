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

#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

#include "python/neml2/types.h"
#include "neml2/models/Model.h"
#include "neml2/base/Factory.h"

namespace py = pybind11;
using namespace neml2;

std::map<VariableName, Tensor>
unpack_model_input(const Model & model, py::dict pyinputs)
{
  std::vector<VariableName> input_names;
  std::vector<Tensor> input_values;
  for (auto && [key, val] : pyinputs)
  {
    try
    {
      input_names.push_back(key.cast<VariableName>());
    }
    catch (py::cast_error &)
    {
      throw py::cast_error("neml2.Model.value: Invalid input key type -- dictionary "
                           "keys must be convertible to neml2.VariableName");
    }

    try
    {
      input_values.push_back(val.cast<Tensor>());
    }
    catch (py::cast_error &)
    {
      if (THPVariable_Check(val.ptr()))
      {
        const auto x = THPVariable_Unpack(val.ptr());
        const auto & xvar = model.input_variable(input_names.back());
        const auto batch_dim = x.dim() - xvar.list_dim() - xvar.base_dim();
        input_values.push_back(Tensor(x, batch_dim));
      }
      else
        throw py::cast_error(
            "neml2.Model.value: Invalid input value type -- dictionary values must "
            "be neml2.Tensor or torch.Tensor");
    }
  }

  std::map<VariableName, Tensor> inputs;
  for (size_t i = 0; i < input_names.size(); ++i)
    inputs[input_names[i]] = input_values[i];

  return inputs;
}

PYBIND11_MODULE(base, m)
{
  m.doc() = "NEML2 Python bindings";

  py::module_::import("neml2.tensors");

  // Definitions
  auto model_cls =
      py::class_<Model, std::shared_ptr<Model>>(m, "Model", "A thin wrapper around neml2::Model");
  auto tensor_value_cls =
      py::class_<TensorValueBase>(m,
                                  "TensorValue",
                                  "The interface for working with tensor values (parameters, "
                                  "buffers, etc.) managed by models.");

  // Factory methods
  m.def("load_input", &load_input, py::arg("path"), py::arg("cli_args") = "", R"(
Parse all options from an input file. Note that Previously loaded input options
will be discarded.

:param path:     Path to the input file to be parsed
:parma cli_args: Additional command-line arguments to pass to the parser
)");
  m.def("reload_input", &reload_input, py::arg("path"), py::arg("cli_args") = "", R"(
Similar to base.load_input, except that this function additionally clears the
factory so that previously retrieved models are deleted.

This function is only needed if you load and evaluate models inside a for-loop,
where it is desirable to deallocate models on-the-fly.

:param path:     Path to the input file to be parsed
:param cli_args: Additional command-line arguments to pass to the parser
)");
  m.def("get_model",
        &get_model,
        py::arg("model"),
        py::arg("force_create") = true,
        py::return_value_policy::reference,
        R"(
Create a models.Model from given input options. The input file must have
already been parsed and loaded.

:param model:        Name of the model
:param force_create: Whether to force create the model even if one has already been created
)");
  m.def("load_model",
        &load_model,
        py::arg("path"),
        py::arg("model"),
        py::return_value_policy::reference,
        R"(
A convenient function to load an input file and get a model.

This function is equivalent to calling base.load_input followed by
base.get_model. Note that this convenient function does not support passing
additional command-line arguments and will force the creation of a new
models.Model even if one has already been created. Use base.load_input and
base.get_model if you need finer control over the model creation behavior.

:param path:      Path to the input file to be parsed
:param model:     Name of the model
)");
  m.def("reload_model",
        &reload_model,
        py::arg("path"),
        py::arg("model"),
        py::return_value_policy::reference,
        R"(
Similar to base.load_model, except that this function additionally clears the
factory so that previously retrieved models are deleted.

This function is only needed if you load and evaluate models inside a for-loop,
where it is desirable to deallocate models on-the-fly.

:param path:      Path to the input file to be parsed
:param model:     Name of the model
)");
  m.def(
      "diagnose",
      [](const Model & m) { diagnose(m); },
      py::arg("model"),
      R"(
Diagnose common issues in model setup. Raises a runtime error including all identified issues, if any.

:param model: Model to be diagnosed
)");

  // neml2.base.TensorValue
  tensor_value_cls
      .def(
          "torch",
          [](const TensorValueBase & self) { return torch::Tensor(Tensor(self)); },
          "Convert to a torch.Tensor")
      .def(
          "tensor",
          [](const TensorValueBase & self) { return Tensor(self); },
          "Convert to a tensors.Tensor")
      .def_property_readonly(
          "requires_grad",
          [](const TensorValueBase & self) { return Tensor(self).requires_grad(); },
          "Value of the boolean requires_grad property of the underlying tensor.")
      .def(
          "requires_grad_",
          [](TensorValueBase & self, bool req) { return self.requires_grad_(req); },
          py::arg("req") = true,
          "Set the requires_grad property of the underlying tensor.")
      .def(
          "set_",
          [](TensorValueBase & self, const Tensor & val) { self = val; },
          "Modify the underlying tensor data.")
      .def_property_readonly(
          "grad",
          [](const TensorValueBase & self) { return Tensor(self).grad(); },
          "Retrieve the accumulated vector-Jacobian product after a backward propagation.");

  // neml2.base.Model
  model_cls.def_property_readonly("name", &Model::name, "Name of the model")
      .def_property_readonly("type", &Model::type, "Type of the model")
      .def(
          "input_axis",
          [](const Model & self) { return &self.input_axis(); },
          py::return_value_policy::reference,
          "Input axis of the model. The axis contains information on variable names and their "
          "associated slicing indices.")
      .def(
          "output_axis",
          [](const Model & self) { return &self.output_axis(); },
          py::return_value_policy::reference,
          "Input axis of the model. The axis contains information on variable names and their "
          "associated slicing indices.")
      .def(
          "named_parameters",
          [](Model & self)
          {
            std::map<std::string, TensorValueBase *> params;
            for (auto && [pname, pval] : self.named_parameters())
              params[pname] = &pval;
            return params;
          },
          py::return_value_policy::reference,
          "Get the model parameters. The keys of the returned dictionary are the parameters' "
          "names.")
      .def(
          "named_buffers",
          [](Model & self)
          {
            std::map<std::string, TensorValueBase *> buffers;
            for (auto && [bname, bval] : self.named_buffers())
              buffers[bname] = &bval;
            return buffers;
          },
          py::return_value_policy::reference,
          "Get the model buffers. The keys of the returned dictionary are the buffers' names.")
      .def("__getattr__",
           py::overload_cast<const std::string &>(&Model::get_parameter, py::const_),
           py::return_value_policy::reference,
           "Get a model parameter given its name")
      .def("__setattr__", &Model::set_parameter, "Set the value for a model parameter")
      .def(
          "set_parameters", &Model::set_parameters, "Set the values for multiple model parameters");

  // Forward operator APIs
  model_cls
      .def("value",
           [](Model & self, py::dict pyinputs)
           { return self.value(unpack_model_input(self, pyinputs)); })
      .def("dvalue",
           [](Model & self, py::dict pyinputs)
           { return self.dvalue(unpack_model_input(self, pyinputs)); })
      .def("d2value",
           [](Model & self, py::dict pyinputs)
           { return self.d2value(unpack_model_input(self, pyinputs)); })
      .def("value_and_dvalue",
           [](Model & self, py::dict pyinputs)
           { return self.value_and_dvalue(unpack_model_input(self, pyinputs)); })
      .def("dvalue_and_d2value",
           [](Model & self, py::dict pyinputs)
           { return self.dvalue_and_d2value(unpack_model_input(self, pyinputs)); })
      .def("value_and_dvalue_and_d2value",
           [](Model & self, py::dict pyinputs)
           { return self.value_and_dvalue_and_d2value(unpack_model_input(self, pyinputs)); });
}
