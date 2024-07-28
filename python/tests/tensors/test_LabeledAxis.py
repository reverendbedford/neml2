# Copyright 2024, UChicago Argonne, LLC
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
from pathlib import Path
import neml2


def test_axis_properties():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_LabeledAxis.i", "model")
    input_axis = model.input_axis()
    output_axis = model.output_axis()

    assert input_axis.storage_size() == 8
    assert output_axis.storage_size() == 1

    assert input_axis.nvariable() == 8
    assert output_axis.nvariable() == 1

    assert input_axis.nsubaxis() == 4
    assert output_axis.nsubaxis() == 1


def test_has_variable():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_LabeledAxis.i", "model")
    input_axis = model.input_axis()
    output_axis = model.output_axis()
    assert input_axis.has_variable("forces/t")
    assert input_axis.has_variable("old_forces/t")
    assert input_axis.has_variable("old_state/foo")
    assert input_axis.has_variable("old_state/bar")
    assert input_axis.has_variable("state/foo")
    assert input_axis.has_variable("state/foo_rate")
    assert input_axis.has_variable("state/bar")
    assert input_axis.has_variable("state/bar_rate")
    assert output_axis.has_variable("residual/foo_bar")
    assert not input_axis.has_variable("foo/bar_rate")
    assert not output_axis.has_variable("forces/foo_bar")


def test_has_subaxis():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_LabeledAxis.i", "model")
    input_axis = model.input_axis()
    output_axis = model.output_axis()
    assert input_axis.has_subaxis("state")
    assert input_axis.has_subaxis("old_state")
    assert input_axis.has_subaxis("forces")
    assert input_axis.has_subaxis("old_forces")
    assert output_axis.has_subaxis("residual")
    assert not input_axis.has_subaxis("residual")
    assert not output_axis.has_subaxis("state")


def test_variable_names():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_LabeledAxis.i", "model")
    assert model.input_axis().variable_names() == [
        "forces/t",
        "old_forces/t",
        "old_state/bar",
        "old_state/foo",
        "state/bar",
        "state/bar_rate",
        "state/foo",
        "state/foo_rate",
    ]
    assert model.output_axis().variable_names() == ["residual/foo_bar"]


def test_subaxis_names():
    pwd = Path(__file__).parent
    model = neml2.reload_model(pwd / "test_LabeledAxis.i", "model")
    assert model.input_axis().subaxis_names() == [
        "forces",
        "old_forces",
        "old_state",
        "state",
    ]
    assert model.output_axis().subaxis_names() == ["residual"]
