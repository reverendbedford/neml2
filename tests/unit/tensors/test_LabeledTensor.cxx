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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "neml2/tensors/LabeledTensor.h"
#include "neml2/tensors/tensors.h"

using namespace neml2;

TEST_CASE("LabeledTensor", "[tensors]")
{
  // Batch shape
  TorchShape batch_sizes{2, 5};

  // Setup the Label
  LabeledAxis info1;
  info1.add<SR2>("first").add<Scalar>("second").add<Scalar>("third");
  info1.setup_layout();

  LabeledAxis info2;
  info2.add<Scalar>("first").add<SR2>("second");
  info2.setup_layout();

  SECTION("Empty constructor")
  {
    LabeledMatrix A;
    REQUIRE(!A.tensor().defined());
    REQUIRE(A.axes().size() == 2);
  }

  SECTION("Zero constructor")
  {
    auto zero = torch::tensor(0.0, default_tensor_options());

    LabeledMatrix A(batch_sizes, {&info1, &info2});
    REQUIRE(torch::allclose(A.tensor(), zero));
    REQUIRE(A.axis(0) == info1);
    REQUIRE(A.axis(1) == info2);
    REQUIRE(A.tensor().batch_sizes() == batch_sizes);
    REQUIRE(A.tensor().base_size(0) == info1.storage_size());
    REQUIRE(A.tensor().base_size(1) == info2.storage_size());
  }

  SECTION("view")
  {
    LabeledMatrix A(batch_sizes, {&info1, &info2});
    auto & view1 = A.view<BatchTensor>({"first", "second"});
    auto & view2 = A.view<SR2>({"first", "first"});
    auto & view3 = A.view<BatchTensor>({"first", "first"});

    REQUIRE(view1.raw_value().base_sizes() == TorchShape{6, 6});
    REQUIRE(view2.raw_value().base_sizes() == TorchShape{6, 1});
    REQUIRE(view3.raw_value().base_sizes() == TorchShape{6, 1});

    REQUIRE(view1.value().base_sizes() == TorchShape{6, 6});
    REQUIRE(view2.value().base_sizes() == TorchShape{6});
    REQUIRE(view3.value().base_sizes() == TorchShape{6, 1});

    // Modifying view1 should affect
    //   - A, since view1 is viewing into A
    // and not view2 nor view3
    view1 = BatchTensor::full({6, 6}, 1.0);
    REQUIRE(torch::allclose(view1.raw_value(), torch::tensor(1.0, default_tensor_options())));
    REQUIRE(torch::allclose(view2.raw_value(), torch::tensor(0.0, default_tensor_options())));
    REQUIRE(torch::allclose(view3.raw_value(), torch::tensor(0.0, default_tensor_options())));

    REQUIRE(torch::allclose(view1.value(), torch::tensor(1.0, default_tensor_options())));
    REQUIRE(torch::allclose(view2.value(), torch::tensor(0.0, default_tensor_options())));
    REQUIRE(torch::allclose(view3.value(), torch::tensor(0.0, default_tensor_options())));

    // Modifying view2 should affect
    //   - A, since view1 is viewing into A
    //   - view3, since view2 and view3 are viewing into the same data
    // and not view1
    view2 = BatchTensor::full({6}, 3.0);
    REQUIRE(torch::allclose(view1.raw_value(), torch::tensor(1.0, default_tensor_options())));
    REQUIRE(torch::allclose(view2.raw_value(), torch::tensor(3.0, default_tensor_options())));
    REQUIRE(torch::allclose(view3.raw_value(), torch::tensor(3.0, default_tensor_options())));

    REQUIRE(torch::allclose(view1.value(), torch::tensor(1.0, default_tensor_options())));
    REQUIRE(torch::allclose(view2.value(), torch::tensor(3.0, default_tensor_options())));
    REQUIRE(torch::allclose(view3.value(), torch::tensor(3.0, default_tensor_options())));
  }

  SECTION("clone")
  {
    LabeledMatrix A(batch_sizes, {&info1, &info2});
    auto & view1 = A.view<SR2>({"first", "first"});

    auto B = A.clone();

    REQUIRE(A.tensor().sizes() == B.tensor().sizes());
    REQUIRE(torch::allclose(A.tensor(), B.tensor()));
    REQUIRE(A.axes() == B.axes());
    REQUIRE(B.views().empty());

    // Modifying A should NOT affect B
    view1 = BatchTensor::full({6}, 3.0);
    REQUIRE(torch::allclose(B.tensor(), torch::tensor(0.0, default_tensor_options())));
  }
}
