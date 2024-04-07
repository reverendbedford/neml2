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

#include <nanobind/operators.h>

#include "python/neml2/misc/indexing.h"
#include "python/neml2/misc/types.h"
#include "neml2/tensors/tensors.h"
#include "neml2/tensors/macros.h"

namespace nb = nanobind;

namespace neml2
{

// Forward declarations
template <class Derived>
void def_BatchView(py::module_ & m, const std::string & name);
template <class Derived>
void def_BaseView(py::module_ & m, const std::string & name);
template <class Derived>
void def_BatchTensorBase(py::class_<Derived> & c);

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
  TorchSize dim() const { return _data->batch_dim(); }
  /// Return the batch size
  TorchShapeRef sizes() const { return _data->batch_sizes(); }
  /// Get a batch
  Derived index(TorchSlice indices) const { return _data->batch_index(indices); }
  /// Set a index sliced on the batch dimensions to a value
  void index_put(TorchSlice indices, const torch::Tensor & other)
  {
    _data->batch_index_put(indices, other);
  }
  /// Return a new view of the tensor with values broadcast along the batch dimensions.
  Derived expand(TorchShapeRef batch_size) const { return _data->batch_expand(batch_size); }
  /// Return a new tensor with values broadcast along the batch dimensions.
  Derived expand_copy(TorchShapeRef batch_size) const
  {
    return _data->batch_expand_copy(batch_size);
  }
  /// Unsqueeze a batch dimension
  Derived unsqueeze(TorchSize d) const { return _data->batch_unsqueeze(d); }
  /// Transpose two batch dimensions
  Derived transpose(TorchSize d1, TorchSize d2) const { return _data->batch_transpose(d1, d2); }

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
  TorchSize dim() const { return _data->base_dim(); }
  /// Return the base size
  TorchShapeRef sizes() const { return _data->base_sizes(); }
  /// Get a base
  BatchTensor index(TorchSlice indices) const { return _data->base_index(indices); }
  /// Set a index sliced on the base dimensions to a value
  void index_put(TorchSlice indices, const torch::Tensor & other)
  {
    _data->base_index_put(indices, other);
  }
  /// Return a new view of the tensor with values broadcast along the base dimensions.
  Derived expand(TorchShapeRef base_size) const { return _data->base_expand(base_size); }
  /// Return a new tensor with values broadcast along the base dimensions.
  Derived expand_copy(TorchShapeRef base_size) const { return _data->base_expand_copy(base_size); }
  /// Unsqueeze a base dimension
  Derived unsqueeze(TorchSize d) const { return _data->base_unsqueeze(d); }
  /// Transpose two base dimensions
  Derived transpose(TorchSize d1, TorchSize d2) const { return _data->base_transpose(d1, d2); }
  /// Return the flattened storage needed just for the base indices
  TorchSize storage() const { return _data->base_storage(); }

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
               .def_prop_ro("shape", &BatchView<Derived>::sizes)
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
#define BATCHTENSORBASE_BATCHVIEW_SETITEM(T)                                                       \
  c.def("__setitem__",                                                                             \
        [](BatchView<Derived> * self, TorchSlice index, const T & src)                             \
        { self->index_put(index, src); })                                                          \
      .def("__setitem__",                                                                          \
           [](BatchView<Derived> * self, at::indexing::TensorIndex index, const T & src)           \
           { self->index_put({index}, src); })
  FOR_ALL_BATCHTENSORBASE(BATCHTENSORBASE_BATCHVIEW_SETITEM);
}

template <class Derived>
void
def_BaseView(py::module_ & m, const std::string & name)
{
  auto c = py::class_<BaseView<Derived>>(m, name.c_str())
               .def(py::init<Derived *>())
               .def("dim", &BaseView<Derived>::dim)
               .def_prop_ro("shape", &BaseView<Derived>::sizes)
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
#define BATCHTENSORBASE_BASEVIEW_SETITEM(T)                                                        \
  c.def("__setitem__",                                                                             \
        [](BaseView<Derived> * self, TorchSlice index, const T & src)                              \
        { self->index_put(index, src); })                                                          \
      .def("__setitem__",                                                                          \
           [](BaseView<Derived> * self, at::indexing::TensorIndex index, const T & src)            \
           { self->index_put({index}, src); })
  FOR_ALL_BATCHTENSORBASE(BATCHTENSORBASE_BASEVIEW_SETITEM);
}

template <class Derived>
void
def_BatchTensorBase(py::class_<Derived> & c)
{
  auto classname = c.attr("__name__").template cast<std::string>();

  // Ctors, conversions, accessors etc.
  c.def(py::init<>())
      .def(py::init<const torch::Tensor &, TorchSize>())
      .def(py::init<const Derived &>())
      .def("__str__",
           [classname](const Derived & self)
           {
             return "<" + classname + " of shape " + utils::stringify(self.batch_sizes()) +
                    utils::stringify(self.base_sizes()) + ">";
           })
      .def("__repr__",
           [](const Derived & self)
           {
             return utils::stringify(self) + '\n' +
                    "Batch shape: " + utils::stringify(self.batch_sizes()) + '\n' +
                    " Base shape: " + utils::stringify(self.base_sizes());
           })
      .def("tensor", [](const Derived & self) { return torch::Tensor(self); })
      .def("defined", &Derived::defined)
      .def("batched", &Derived::batched)
      .def("dim", &Derived::dim)
      .def_prop_ro("shape", &Derived::sizes)
      .def("clone", [](Derived * self) { return self->clone(); })
      .def("detach", &Derived::detach)
      .def(
          "to",
          [](Derived * self, NEML2_TENSOR_OPTIONS_VARGS) { return self->to(NEML2_TENSOR_OPTIONS); },
          py::kw_only(),
          PY_ARG_TENSOR_OPTIONS)
      .def_prop_ro("batch", [](Derived * self) { return BatchView<Derived>(self); })
      .def_prop_ro("base", [](Derived * self) { return BaseView<Derived>(self); })
      .def_prop_ro("device", &Derived::device)
      .def_prop_ro("dtype", &Derived::scalar_type);

  // Binary, unary operators
  c.def(float() + py::self)
      .def(py::self + float())
      .def(py::self + py::self)
      .def(float() - py::self)
      .def(py::self - float())
      .def(py::self - py::self)
      .def(float() * py::self)
      .def(py::self * float())
      .def(float() / py::self)
      .def(py::self / float())
      .def(py::self / py::self)
      .def(-py::self)
      .def("__pow__", [](const Derived & a, float b) { return math::pow(a, b); })
      .def("__pow__", [](const Derived & a, const Scalar & b) { return math::pow(a, b); })
      .def("__rpow__", [](const Derived & b, float a) { return math::pow(a, b); })
      .def("__pow__", [](const Derived & a, const Derived & b) { return math::pow(a, b); });

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
