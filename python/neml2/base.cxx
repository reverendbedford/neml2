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

#include "python/neml2/types.h"
#include "neml2/models/Model.h"
#include "neml2/base/Factory.h"

namespace py = pybind11;
using namespace neml2;

PYBIND11_MODULE(base, m)
{
  m.doc() = "NEML2 Python bindings";

  py::module_::import("neml2.tensors");

  // Define neml2.base.Model
  auto model_cls = py::class_<Model, std::shared_ptr<Model>>(m, "Model");

  // Factory methods
  m.def("load_input", &load_input, py::arg("path"), py::arg("cli_args") = "");
  m.def("reload_input", &reload_input, py::arg("path"), py::arg("cli_args") = "");
  m.def("get_model",
        &get_model,
        py::arg("model"),
        py::arg("enable_AD") = true,
        py::arg("force_create") = true);
  m.def("load_model", &load_model, py::arg("path"), py::arg("model"), py::arg("enable_AD") = true);
  m.def("reload_model",
        &reload_model,
        py::arg("path"),
        py::arg("model"),
        py::arg("enable_AD") = true);

  // neml2.base.Model methods
  model_cls.def_property_readonly("is_AD_enabled", &Model::is_AD_enabled)
      .def_property_readonly("is_AD_disabled", &Model::is_AD_disabled)
      .def("reinit",
           py::overload_cast<TensorShapeRef, int, const torch::Device &, const torch::Dtype &>(
               &Model::reinit),
           py::arg("batch_shape"),
           py::arg("deriv_order") = 0,
           py::arg("device") = torch::Device(default_device()),
           py::arg("dtype") = torch::Dtype(default_dtype()))
      .def(
          "input_axis",
          [](const Model & self) { return &self.input_axis(); },
          py::return_value_policy::reference)
      .def(
          "output_axis",
          [](const Model & self) { return &self.output_axis(); },
          py::return_value_policy::reference)
      .def("input_type", &Model::input_type, py::arg("variable"))
      .def("output_type", &Model::output_type, py::arg("variable"))
      .def("value",
           [](Model & self, py::object x)
           {
             // Check if it is a torch.Tensor, if it is, we can wrap it as a LabeledVector since we
             // know a LabeledVector has base_dim == 1.
             if (THPVariable_Check(x.ptr()))
             {
               auto & x_torch = THPVariable_Unpack(x.ptr());
               auto y = self.value(LabeledVector(x_torch, x_torch.dim() - 1, {&self.input_axis()}));
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
           })
      .def("value_and_dvalue",
           [](Model & self, py::object x)
           {
             // Check if it is a torch.Tensor, if it is, we can wrap it as a LabeledVector since we
             // know a LabeledVector has base_dim == 1.
             if (THPVariable_Check(x.ptr()))
             {
               auto & x_torch = THPVariable_Unpack(x.ptr());
               auto [y, dy_dx] = self.value_and_dvalue(
                   LabeledVector(x_torch, x_torch.dim() - 1, {&self.input_axis()}));
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
           })
      .def("named_parameters",
           [](Model & self)
           {
             std::map<std::string, Tensor> params;
             for (auto && [pname, pval] : self.named_parameters())
               params[utils::stringify(pname)] = Tensor(pval);
             return params;
           })
      .def("named_buffers",
           [](Model & self)
           {
             std::map<std::string, Tensor> buffers;
             for (auto && [bname, bval] : self.named_buffers())
               buffers[utils::stringify(bname)] = Tensor(bval);
             return buffers;
           });
}
