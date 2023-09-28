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

#include <catch2/catch.hpp>

#include "neml2/tensors/LabeledTensor.h"

using namespace neml2;

TEST_CASE("LabeledTensor", "[tensors]")
{
  SECTION("operator()")
  {
    TorchSize nbatch = 10;

    // Setup the Label
    LabeledAxis info1;
    info1.add<SR2>("first").add<Scalar>("second").add<Scalar>("third");
    info1.setup_layout();

    LabeledAxis info2;
    info2.add<Scalar>("first").add<SR2>("second");
    info2.setup_layout();

    SECTION("logically 1D LabeledTensor")
    {
      auto A = LabeledTensor<1>::zeros(nbatch, {&info1});
      REQUIRE(A("first").sizes() == TorchShapeRef({nbatch, 6}));
      REQUIRE(A("second").sizes() == TorchShapeRef({nbatch, 1}));
      REQUIRE(A("third").sizes() == TorchShapeRef({nbatch, 1}));
    }

    SECTION("logically 2D LabeledTensor")
    {
      auto A = LabeledTensor<2>::zeros(nbatch, {&info1, &info2});
      REQUIRE(A("first", "first").sizes() == TorchShapeRef({nbatch, 6, 1}));
      REQUIRE(A("first", "second").sizes() == TorchShapeRef({nbatch, 6, 6}));
      REQUIRE(A("second", "first").sizes() == TorchShapeRef({nbatch, 1, 1}));
      REQUIRE(A("second", "second").sizes() == TorchShapeRef({nbatch, 1, 6}));
      REQUIRE(A("third", "first").sizes() == TorchShapeRef({nbatch, 1, 1}));
      REQUIRE(A("third", "second").sizes() == TorchShapeRef({nbatch, 1, 6}));
    }
  }

  SECTION("get")
  {
    TorchSize nbatch = 10;

    // Setup the Label
    LabeledAxis info1;
    info1.add<SR2>("first").add<Scalar>("second").add<Scalar>("third");
    info1.setup_layout();

    LabeledAxis info2;
    info2.add<Scalar>("first").add<SR2>("second");
    info2.setup_layout();

    SECTION("logically 1D LabeledTensor")
    {
      auto A = LabeledTensor<1>::zeros(nbatch, {&info1});
      REQUIRE(A.get<SR2>("first").sizes() == TorchShapeRef({nbatch, 6}));
      REQUIRE(A.get<Scalar>("second").sizes() == TorchShapeRef({nbatch}));
      REQUIRE(A.get<Scalar>("third").sizes() == TorchShapeRef({nbatch}));
    }

    SECTION("logically 2D LabeledTensor")
    {
      auto A = LabeledTensor<2>::zeros(nbatch, {&info1, &info2});
      REQUIRE(A.get<SR2>("first", "first").sizes() == TorchShapeRef({nbatch, 6}));
      REQUIRE(A.get<Scalar>("second", "first").sizes() == TorchShapeRef({nbatch}));
      REQUIRE(A.get<SR2>("second", "second").sizes() == TorchShapeRef({nbatch, 6}));
      REQUIRE(A.get<Scalar>("third", "first").sizes() == TorchShapeRef({nbatch}));
      REQUIRE(A.get<SR2>("third", "second").sizes() == TorchShapeRef({nbatch, 6}));
    }
  }

  SECTION("set")
  {
    TorchSize nbatch = 10;

    // Setup the Label
    LabeledAxis info1;
    info1.add<SR2>("first").add<Scalar>("second").add<Scalar>("third");
    info1.setup_layout();

    LabeledAxis info2;
    info2.add<Scalar>("first").add<SR2>("second");
    info2.setup_layout();

    SECTION("logically 1D LabeledTensor")
    {
      auto A = LabeledTensor<1>::zeros(nbatch, {&info1});
      A.set(BatchTensor::ones(6).batch_expand(nbatch), "first");
      REQUIRE(torch::sum(A("first")).item<double>() == Approx(nbatch * 6));
      REQUIRE(torch::sum(A("second")).item<double>() == Approx(0));
      REQUIRE(torch::sum(A("third")).item<double>() == Approx(0));
    }

    SECTION("logically 2D LabeledTensor")
    {
      auto A = LabeledTensor<2>::zeros(nbatch, {&info1, &info2});
      A.set(BatchTensor::ones({1, 6}).batch_expand(nbatch), "third", "second");
      REQUIRE(torch::sum(A("first", "first")).item<double>() == Approx(0));
      REQUIRE(torch::sum(A("first", "second")).item<double>() == Approx(0));
      REQUIRE(torch::sum(A("second", "first")).item<double>() == Approx(0));
      REQUIRE(torch::sum(A("second", "second")).item<double>() == Approx(0));
      REQUIRE(torch::sum(A("third", "first")).item<double>() == Approx(0));
      REQUIRE(torch::sum(A("third", "second")).item<double>() == Approx(nbatch * 6));
    }
  }

  SECTION("clone")
  {
    TorchSize nbatch = 10;

    // Setup the Label
    LabeledAxis info1;
    info1.add<SR2>("first").add<Scalar>("second").add<Scalar>("third");
    info1.setup_layout();

    LabeledAxis info2;
    info2.add<Scalar>("first").add<SR2>("second");
    info2.setup_layout();

    auto A = LabeledTensor<2>::zeros(nbatch, {&info1, &info2});
    auto B = A.clone();

    REQUIRE(A.axis(0) == B.axis(0));
    REQUIRE(A.axis(1) == B.axis(1));
    REQUIRE(torch::allclose(A.tensor(), B.tensor()));

    // Since B is a deep copy, modifying B shouldn't affect A.
    B.set(BatchTensor::ones({1, 6}).batch_expand(nbatch), "third", "second");
    REQUIRE(torch::sum(A("third", "second")).item<double>() == Approx(0));
    REQUIRE(torch::sum(B("third", "second")).item<double>() == Approx(nbatch * 6));
  }

  SECTION("slice")
  {
    TorchSize nbatch = 10;

    // Setup the Label
    LabeledAxis info1;
    info1.add<SR2>("first").add<Scalar>("second").add<Scalar>("third");
    info1.add<LabeledAxis>("sub1").add<LabeledAxis>("sub2");
    info1.subaxis("sub1").add<SR2>("first").add<Scalar>("second");
    info1.subaxis("sub2").add<Scalar>("first").add<Scalar>("second");
    info1.setup_layout();

    LabeledAxis info2;
    info2.add<Scalar>("first").add<SR2>("second");
    info2.setup_layout();

    SECTION("logically 1D LabeledTensor")
    {
      auto A = LabeledTensor<1>::zeros(nbatch, {&info1});
      A.set(2.3 * BatchTensor::ones(7).batch_expand(nbatch), "sub1");
      auto B = A.slice(0, "sub1");
      REQUIRE(torch::sum(B("first")).item<double>() == Approx(nbatch * 6 * 2.3));
      REQUIRE(torch::sum(B("second")).item<double>() == Approx(nbatch * 2.3));
    }

    SECTION("logically 2D LabeledTensor")
    {
      auto A = LabeledTensor<2>::zeros(nbatch, {&info1, &info2});
      A.set(-1.9 * BatchTensor::ones({7, 6}).batch_expand(nbatch), "sub1", "second");
      auto B = A.slice(0, "sub1");
      REQUIRE(torch::sum(B("first", "first")).item<double>() == Approx(0));
      REQUIRE(torch::sum(B("first", "second")).item<double>() == Approx(nbatch * 6 * 6 * -1.9));
      REQUIRE(torch::sum(B("second", "first")).item<double>() == Approx(0));
      REQUIRE(torch::sum(B("second", "second")).item<double>() == Approx(nbatch * 1 * 6 * -1.9));
    }
  }
}
