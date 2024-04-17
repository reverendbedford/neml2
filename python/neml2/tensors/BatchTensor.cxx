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

#include "python/neml2/tensors/BatchTensorBase.h"

#include "neml2/tensors/macros.h"

namespace py = pybind11;
using namespace neml2;

void
def_BatchTensor(py::class_<BatchTensor> & c)
{
  // All FixedDimTensors are convertible to BatchTensor
#define BATCHTENSOR_FROM_FIXEDDIMTENSOR(T) c.def(py::init<T>());
  FOR_ALL_FIXEDDIMTENSOR(BATCHTENSOR_FROM_FIXEDDIMTENSOR);

  // Static methods
  c.def_static(
       "empty",
       [](const TorchShapeRef & base_shape, NEML2_TENSOR_OPTIONS_VARGS)
       { return BatchTensor::empty(base_shape, NEML2_TENSOR_OPTIONS); },
       py::arg("base_shape"),
       py::kw_only(),
       PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "empty",
          [](const TorchShapeRef & batch_shape,
             const TorchShapeRef & base_shape,
             NEML2_TENSOR_OPTIONS_VARGS)
          { return BatchTensor::empty(batch_shape, base_shape, NEML2_TENSOR_OPTIONS); },
          py::arg("batch_shape"),
          py::arg("base_shape"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "zeros",
          [](const TorchShapeRef & base_shape, NEML2_TENSOR_OPTIONS_VARGS)
          { return BatchTensor::zeros(base_shape, NEML2_TENSOR_OPTIONS); },
          py::arg("base_shape"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "zeros",
          [](const TorchShapeRef & batch_shape,
             const TorchShapeRef & base_shape,
             NEML2_TENSOR_OPTIONS_VARGS)
          { return BatchTensor::zeros(batch_shape, base_shape, NEML2_TENSOR_OPTIONS); },
          py::arg("batch_shape"),
          py::arg("base_shape"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "ones",
          [](const TorchShapeRef & base_shape, NEML2_TENSOR_OPTIONS_VARGS)
          { return BatchTensor::ones(base_shape, NEML2_TENSOR_OPTIONS); },
          py::arg("base_shape"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "ones",
          [](const TorchShapeRef & batch_shape,
             const TorchShapeRef & base_shape,
             NEML2_TENSOR_OPTIONS_VARGS)
          { return BatchTensor::ones(batch_shape, base_shape, NEML2_TENSOR_OPTIONS); },
          py::arg("batch_shape"),
          py::arg("base_shape"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "full",
          [](const TorchShapeRef & base_shape, Real init, NEML2_TENSOR_OPTIONS_VARGS)
          { return BatchTensor::full(base_shape, init, NEML2_TENSOR_OPTIONS); },
          py::arg("base_shape"),
          py::arg("fill_value"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "full",
          [](const TorchShapeRef & batch_shape,
             const TorchShapeRef & base_shape,
             Real init,
             NEML2_TENSOR_OPTIONS_VARGS)
          { return BatchTensor::full(batch_shape, base_shape, init, NEML2_TENSOR_OPTIONS); },
          py::arg("batch_shape"),
          py::arg("base_shape"),
          py::arg("fill_value"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "identity",
          [](TorchSize n, NEML2_TENSOR_OPTIONS_VARGS)
          { return BatchTensor::identity(n, NEML2_TENSOR_OPTIONS); },
          py::arg("n"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_static(
          "identity",
          [](const TorchShapeRef & batch_shape, TorchSize n, NEML2_TENSOR_OPTIONS_VARGS)
          { return BatchTensor::identity(batch_shape, n, NEML2_TENSOR_OPTIONS); },
          py::arg("batch_shape"),
          py::arg("n"),
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS);
}
