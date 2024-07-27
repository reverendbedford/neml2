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

#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

#include "python/neml2/types.h"
#include "neml2/models/Model.h"
#include "neml2/base/Factory.h"

namespace py = pybind11;
using namespace neml2;

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
        py::arg("enable_AD") = true,
        py::arg("force_create") = true,
        py::return_value_policy::reference,
        R"(
Create a models.Model from given input options. The input file must have
already been parsed and loaded.

:param model:        Name of the model
:param enable_AD:    Enable automatic differentiation
:param force_create: Whether to force create the model even if one has already been created
)");
  m.def("load_model",
        &load_model,
        py::arg("path"),
        py::arg("model"),
        py::arg("enable_AD") = true,
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
:param enable_AD: Enable automatic differentiation
)");
  m.def("reload_model",
        &reload_model,
        py::arg("path"),
        py::arg("model"),
        py::arg("enable_AD") = true,
        py::return_value_policy::reference,
        R"(
Similar to base.load_model, except that this function additionally clears the
factory so that previously retrieved models are deleted.

This function is only needed if you load and evaluate models inside a for-loop,
where it is desirable to deallocate models on-the-fly.

:param path:      Path to the input file to be parsed
:param model:     Name of the model
:param enable_AD: Enable automatic differentiation
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
          [](const TensorValueBase & self) { return torch::Tensor(Tensor(self)).clone(); },
          R"(
Convert this to a torch.Tensor.

This conversion takes ownership of the tensor, and so any modification to the
returned torch.Tensor does not affect the original tensors.TensorValue. Use
tensors.TensorValue.set_ instead to modify the tensor value.
)")
      .def(
          "tensor",
          [](const TensorValueBase & self) { return Tensor(self).clone(); },
          R"(
Convert this to a tensors.Tensor.

This conversion takes ownership of the tensor, and so any modification to the
returned tensors.Tensor does not affect the original tensors.TensorValue. Use
tensors.TensorValue.set_ instead to modify the tensor value.
)")
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
      .def_property_readonly(
          "is_AD_enabled",
          &Model::is_AD_enabled,
          "Whether automatic differentiation is enabled for this model. This property cannot be "
          "modified once the model is created. Use the `enable_AD` option of base.load_model or "
          "base.get_model to control this property.")
      .def("reinit",
           py::overload_cast<TensorShapeRef, int, const torch::Device &, const torch::Dtype &>(
               &Model::reinit),
           py::arg("batch_shape") = TensorShapeRef{},
           py::arg("deriv_order") = 0,
           py::arg("device") = torch::Device(default_device()),
           py::arg("dtype") = torch::Dtype(default_dtype()),
           R"(
(Re)initialize the model with given batch shape, derivative order, device, and dtype.

:param batch_shape: Batch shape used to allocate input, output, and derivative storage
:param deriv_order: An integer ranging from 0-2. When set to 0, only the output storage
    will be allocated; when set to 1, both the output and the first derivative storage
    are allocated; when set to 2, the second derivative storage is additionally allocated.
:param device:      Device on which the model will be evaluated. All parameters, buffers,
    and custom data are synced to the given device.
:param dtype:       Floating point scalar type used throughout the model.
)")
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
          "input_type",
          &Model::input_type,
          py::arg("variable"),
          "Introspect the underlying tensor type of an input variable. @returns tensors.TensorType")
      .def("output_type",
           &Model::output_type,
           py::arg("variable"),
           "Introspect the underlying tensor type of an output variable. @returns "
           "tensors.TensorType")
      .def(
          "value",
          [](Model & self, py::object x)
          {
            // Check if it is a torch.Tensor, if it is, we can wrap it as a LabeledVector since we
            // know a LabeledVector has base_dim == 1.
            if (THPVariable_Check(x.ptr()))
            {
              auto & x_torch = THPVariable_Unpack(x.ptr());
              auto y = self.value(LabeledVector(x_torch, {&self.input_axis()}));
              return py::cast(torch::Tensor(y));
            }

            // Check if it is a neml2.Tensor
            try
            {
              auto y = self.value(LabeledVector(x.cast<Tensor>(), {&self.input_axis()}));
              return py::cast(y.tensor());
            }
            catch (py::cast_error &)
            {
            }

            // Otherwise, hope it is a LabeledVector
            return py::cast(self.value(x.cast<LabeledVector>()));
          },
          "Evaluate the model with given input and return the output. Note that the input can "
          "either be of type torch.Tensor, tensors.Tensor, or tensors.LabeledVector. The returned "
          "output will be of the covariant type of the input.")
      .def(
          "value_and_dvalue",
          [](Model & self, py::object x)
          {
            // Check if it is a torch.Tensor, if it is, we can wrap it as a LabeledVector since we
            // know a LabeledVector has base_dim == 1.
            if (THPVariable_Check(x.ptr()))
            {
              auto & x_torch = THPVariable_Unpack(x.ptr());
              auto [y, dy_dx] = self.value_and_dvalue(LabeledVector(x_torch, {&self.input_axis()}));
              return py::make_tuple(torch::Tensor(y), torch::Tensor(dy_dx));
            }

            // Check if it is a neml2.Tensor
            try
            {
              auto [y, dy_dx] =
                  self.value_and_dvalue(LabeledVector(x.cast<Tensor>(), {&self.input_axis()}));
              return py::make_tuple(y.tensor(), dy_dx.tensor());
            }
            catch (py::cast_error &)
            {
            }

            // Otherwise, hope it is a LabeledVector
            auto [y, dy_dx] = self.value_and_dvalue(x.cast<LabeledVector>());
            return py::make_tuple(y, dy_dx);
          },
          "Evaluate the model with given input and return the output as well the first derivative. "
          "Note that the input can either be of type torch.Tensor, tensors.Tensor, or "
          "tensors.LabeledVector. The returned output and derivative will be of the covariant type "
          "of the input.")
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
      .def(
          "named_submodels",
          [](const Model & self)
          {
            std::map<std::string, Model *> submodels;
            for (auto submodel : self.registered_models())
              submodels[submodel->name()] = submodel;
            return submodels;
          },
          py::return_value_policy::reference,
          "Get the sub-models registered to this model")
      .def("get_parameter",
           &Model::get_parameter,
           py::return_value_policy::reference,
           "Get a model parameter given its name")
      .def("set_parameter", &Model::set_parameter, "Set the value for a model parameter")
      .def("set_parameters", &Model::set_parameters, "Set the values for multiple model parameters")
      .def(
          "dependency",
          [](const Model & self)
          {
            std::map<std::string, const Model *> deps;
            for (auto && [name, var] : self.input_variables())
              if (var.src())
                deps[utils::stringify(name)] = &var.src()->owner();
            return deps;
          },
          py::return_value_policy::reference,
          "Get the dictionary describing this model's dependency information, if any.");
}
