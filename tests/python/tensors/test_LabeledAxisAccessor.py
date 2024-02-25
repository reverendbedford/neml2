# Copyright 2023, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: NEML2 -- the New Engineering material Model Library, version 2
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest

from neml2.tensors import LabeledAxisAccessor


@pytest.mark.it("Constructors")
def test_ctors():
    A = LabeledAxisAccessor()
    assert str(A) == ""

    A = LabeledAxisAccessor("state")
    assert str(A) == "state"

    A = LabeledAxisAccessor("force")
    assert str(A) == "force"

    A = LabeledAxisAccessor("state", "stress")
    assert str(A) == "state/stress"

    A = LabeledAxisAccessor("state", "internal", "gamma")
    assert str(A) == "state/internal/gamma"


@pytest.mark.it("empty")
def test_empty():
    A = LabeledAxisAccessor()
    B = LabeledAxisAccessor("state", "stress")
    assert A.empty()
    assert not B.empty()


@pytest.mark.it("size")
def test_size():
    A = LabeledAxisAccessor()
    B = LabeledAxisAccessor("state", "stress")
    assert A.size() == 0
    assert B.size() == 2


@pytest.mark.it("with_suffix")
def test_with_suffix():
    A = LabeledAxisAccessor("state", "stress")
    B = LabeledAxisAccessor("state", "stress_foo")
    assert A.with_suffix("_foo") == B


@pytest.mark.it("append")
def test_append():
    A = LabeledAxisAccessor("state")
    B = LabeledAxisAccessor("foo", "bar")
    C = LabeledAxisAccessor("state", "foo", "bar")
    assert A.append(B) == C


@pytest.mark.it("on")
def test_on():
    A = LabeledAxisAccessor("stress")
    B = LabeledAxisAccessor("residual", "stress")
    assert A.on(LabeledAxisAccessor("residual")) == B


@pytest.mark.it("start_with")
def test_start_with():
    A = LabeledAxisAccessor("internal", "stress", "foo")
    B = LabeledAxisAccessor("internal", "stress")
    C = LabeledAxisAccessor("residual", "stress")
    assert A.start_with(B)
    assert not A.start_with(C)
