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


mkdir -p coverage
export ROOT=$(pwd)
export INCLUDE_DIR=$ROOT/include
export SRC_DIR=$ROOT/src
# Let's not worry about the coverage of test utils for now
# export TEST_INCLUDE_DIR=$ROOT/tests/include
# export TEST_SRC_DIR=$ROOT/tests/src
export COVERAGE_DIR=$ROOT/coverage
lcov --gcov-tool gcov --capture --initial --directory $SRC_DIR --output-file $COVERAGE_DIR/initialize.info
cd tests
./unit/unit_tests || true
cd ..
lcov --gcov-tool gcov --capture --ignore-errors gcov,source --directory $SRC_DIR --output-file $COVERAGE_DIR/covered.info
lcov --gcov-tool gcov --add-tracefile $COVERAGE_DIR/initialize.info --add-tracefile $COVERAGE_DIR/covered.info --output-file $COVERAGE_DIR/final.info
lcov --gcov-tool gcov --extract $COVERAGE_DIR/final.info \*$SRC_DIR/\* --extract $COVERAGE_DIR/final.info \*$INCLUDE_DIR/\* --output-file $COVERAGE_DIR/coverage.info
genhtml $COVERAGE_DIR/coverage.info --output-directory $COVERAGE_DIR > $COVERAGE_DIR/genhtml.out
