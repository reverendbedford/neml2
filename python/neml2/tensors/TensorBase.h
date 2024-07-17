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

#pragma once

#include <pybind11/operators.h>

#include "python/neml2/indexing.h"
#include "python/neml2/types.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/macros.h"
#include "neml2/misc/math.h"

namespace py = pybind11;

namespace neml2
{

// Forward declarations
template <class Derived>
void def_BatchView(py::module_ & m, const std::string & name);
template <class Derived>
void def_BaseView(py::module_ & m, const std::string & name);
template <class Derived>
void def_TensorBase(py::class_<Derived> & c);

/**
 * @brief Convenient shim wrapper for working with batch dimensions
 *
 * The view does NOT extend the life of of the wrapped tensor.
 */
template <class Derived>
class BatchView
{
public:
  BatchView(Derived * data)
    : _data(data)
  {
  }
  /// Return the number of batch dimensions
  Size dim() const { return _data->batch_dim(); }
  /// Return the batch size
  TensorShapeRef sizes() const { return _data->batch_sizes(); }
  /// Get a batch
  Derived index(indexing::TensorIndices indices) const { return _data->batch_index(indices); }
  /// Set a index sliced on the batch dimensions to a value
  void index_put(indexing::TensorIndices indices, const torch::Tensor & other)
  {
    _data->batch_index_put(indices, other);
  }
  /// Return a new view of the tensor with values broadcast along the batch dimensions.
  Derived expand(TensorShapeRef batch_size) const { return _data->batch_expand(batch_size); }
  /// Return a new tensor with values broadcast along the batch dimensions.
  Derived expand_copy(TensorShapeRef batch_size) const
  {
    return _data->batch_expand_copy(batch_size);
  }
  /// Unsqueeze a batch dimension
  Derived unsqueeze(Size d) const { return _data->batch_unsqueeze(d); }
  /// Transpose two batch dimensions
  Derived transpose(Size d1, Size d2) const { return _data->batch_transpose(d1, d2); }

private:
  Derived * _data;
};

/**
 * @brief Convenient shim wrapper for working with base dimensions
 *
 * The view does NOT extend the life of of the wrapped tensor.
 */
template <class Derived>
class BaseView
{
public:
  BaseView(Derived * data)
    : _data(data)
  {
  }
  /// Return the number of base dimensions
  Size dim() const { return _data->base_dim(); }
  /// Return the base size
  TensorShapeRef sizes() const { return _data->base_sizes(); }
  /// Get a base
  Tensor index(indexing::TensorIndices indices) const { return _data->base_index(indices); }
  /// Set a index sliced on the base dimensions to a value
  void index_put(indexing::TensorIndices indices, const torch::Tensor & other)
  {
    _data->base_index_put(indices, other);
  }
  /// Return a new view of the tensor with values broadcast along the base dimensions.
  Derived expand(TensorShapeRef base_size) const { return _data->base_expand(base_size); }
  /// Return a new tensor with values broadcast along the base dimensions.
  Derived expand_copy(TensorShapeRef base_size) const { return _data->base_expand_copy(base_size); }
  /// Unsqueeze a base dimension
  Derived unsqueeze(Size d) const { return _data->base_unsqueeze(d); }
  /// Transpose two base dimensions
  Derived transpose(Size d1, Size d2) const { return _data->base_transpose(d1, d2); }
  /// Return the flattened storage needed just for the base indices
  Size storage() const { return _data->base_storage(); }

private:
  Derived * _data;
};
} // namespace neml2

///////////////////////////////////////////////////////////////////////////////
// Implementations
///////////////////////////////////////////////////////////////////////////////
namespace neml2
{
template <class Derived>
void
def_BatchView(py::module_ & m, const std::string & name)
{
  auto c = py::class_<BatchView<Derived>>(m, name.c_str())
               .def(py::init<Derived *>())
               .def("dim", &BatchView<Derived>::dim)
               .def_property_readonly("shape", &BatchView<Derived>::sizes)
               .def("__getitem__", &BatchView<Derived>::index)
               .def("__getitem__",
                    [](BatchView<Derived> * self, at::indexing::TensorIndex index)
                    { return self->index({index}); })
               .def("__setitem__", &BatchView<Derived>::index_put)
               .def("__setitem__",
                    [](BatchView<Derived> * self,
                       at::indexing::TensorIndex index,
                       const torch::Tensor & src) { return self->index_put({index}, src); })
               .def("expand", &BatchView<Derived>::expand)
               .def("expand_copy", &BatchView<Derived>::expand_copy)
               .def("unsqueeze", &BatchView<Derived>::unsqueeze)
               .def("transpose", &BatchView<Derived>::transpose);

  // The setter should also take any primitive tensor type
#define TENSORBASE_BATCHVIEW_SETITEM(T)                                                            \
  c.def("__setitem__",                                                                             \
        [](BatchView<Derived> * self, indexing::TensorIndices index, const T & src)                \
        { self->index_put(index, src); })                                                          \
      .def("__setitem__",                                                                          \
           [](BatchView<Derived> * self, at::indexing::TensorIndex index, const T & src)           \
           { self->index_put({index}, src); })
  FOR_ALL_TENSORBASE(TENSORBASE_BATCHVIEW_SETITEM);
}

template <class Derived>
void
def_BaseView(py::module_ & m, const std::string & name)
{
  auto c = py::class_<BaseView<Derived>>(m, name.c_str())
               .def(py::init<Derived *>())
               .def("dim", &BaseView<Derived>::dim)
               .def_property_readonly("shape", &BaseView<Derived>::sizes)
               .def("__getitem__", &BaseView<Derived>::index)
               .def("__getitem__",
                    [](BaseView<Derived> * self, at::indexing::TensorIndex index)
                    { return self->index({index}); })
               .def("__setitem__", &BaseView<Derived>::index_put)
               .def("__setitem__",
                    [](BaseView<Derived> * self,
                       at::indexing::TensorIndex index,
                       const torch::Tensor & src) { self->index_put({index}, src); })
               .def("expand", &BaseView<Derived>::expand)
               .def("expand_copy", &BaseView<Derived>::expand_copy)
               .def("unsqueeze", &BaseView<Derived>::unsqueeze)
               .def("transpose", &BaseView<Derived>::transpose)
               .def("storage", &BaseView<Derived>::storage);

  // The setter should also take any primitive tensor type
#define TENSORBASE_BASEVIEW_SETITEM(T)                                                             \
  c.def("__setitem__",                                                                             \
        [](BaseView<Derived> * self, indexing::TensorIndices index, const T & src)                 \
        { self->index_put(index, src); })                                                          \
      .def("__setitem__",                                                                          \
           [](BaseView<Derived> * self, at::indexing::TensorIndex index, const T & src)            \
           { self->index_put({index}, src); })
  FOR_ALL_TENSORBASE(TENSORBASE_BASEVIEW_SETITEM);
}

template <class Derived>
void
def_TensorBase(py::class_<Derived> & c)
{
  auto classname = c.attr("__name__").template cast<std::string>();

  // Ctors, conversions, accessors etc.
  c.def(py::init<>())
      .def(py::init<const torch::Tensor &, Size>())
      .def(py::init<const Derived &>())
      .def("__str__",
           [classname](const Derived & self)
           {
             return utils::stringify(self) + '\n' + "<" + classname + " of shape " +
                    utils::stringify(self.batch_sizes()) + utils::stringify(self.base_sizes()) +
                    ">";
           })
      .def("__repr__",
           [classname](const Derived & self)
           {
             return "<" + classname + " of shape " + utils::stringify(self.batch_sizes()) +
                    utils::stringify(self.base_sizes()) + ">";
           })
      .def_property_readonly("batch", [](Derived * self) { return new BatchView<Derived>(self); })
      .def_property_readonly("base", [](Derived * self) { return new BaseView<Derived>(self); })
      .def("clone", [](Derived * self) { return self->clone(); })
      .def("detach", &Derived::detach)
      .def(
          "to",
          [](Derived * self, NEML2_TENSOR_OPTIONS_VARGS) { return self->to(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def("torch", [](const Derived & self) { return torch::Tensor(self); })
      .def("defined", &Derived::defined)
      .def("batched", &Derived::batched)
      .def("dim", &Derived::dim)
      .def_property_readonly("shape", &Derived::sizes)
      .def_property_readonly("dtype", &Derived::scalar_type)
      .def_property_readonly("device", &Derived::device)
      .def("requires_grad_", &Derived::requires_grad_)
      .def_property_readonly("requires_grad", &Derived::requires_grad)
      .def_property_readonly("grad", &Derived::grad);

  // Binary, unary operators
  c.def(float() + py::self)
      .def(py::self + float())
      .def(py::self + Scalar())
      .def(py::self + py::self)
      .def(float() - py::self)
      .def(py::self - float())
      .def(py::self - Scalar())
      .def(py::self - py::self)
      .def(float() * py::self)
      .def(py::self * float())
      .def(py::self * Scalar())
      .def(float() / py::self)
      .def(py::self / float())
      .def(py::self / Scalar())
      .def(py::self / py::self)
      .def(-py::self)
      .def("__pow__", [](const Derived & a, float b) { return math::pow(a, b); })
      .def("__pow__", [](const Derived & a, const Scalar & b) { return math::pow(a, b); })
      .def("__rpow__", [](const Derived & a, float b) { return math::pow(b, a); });

  // Static methods
  c.def_static("empty_like", &Derived::empty_like)
      .def_static("zeros_like", &Derived::zeros_like)
      .def_static("ones_like", &Derived::ones_like)
      .def_static("full_like", &Derived::full_like)
      .def_static("linspace",
                  &Derived::linspace,
                  py::arg("start"),
                  py::arg("end"),
                  py::arg("nstep"),
                  py::arg("dim") = 0,
                  py::arg("batch_dim") = -1)
      .def_static("logspace",
                  &Derived::logspace,
                  py::arg("start"),
                  py::arg("end"),
                  py::arg("nstep"),
                  py::arg("dim") = 0,
                  py::arg("batch_dim") = -1,
                  py::arg("base") = 10.0);
}
} // namespace neml2
