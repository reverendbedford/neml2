mkdir -p coverage
export ROOT=$(pwd)
export SRC_DIR=$ROOT/src
export INCLUDE_DIR=$ROOT/include
export COVERAGE_DIR=$ROOT/coverage
lcov --gcov-tool gcov --capture --initial --directory $SRC_DIR --output-file $COVERAGE_DIR/initialize.info
./tests/tests || true
lcov --gcov-tool gcov --capture --ignore-errors gcov,source --directory $SRC_DIR --output-file $COVERAGE_DIR/covered.info
lcov --gcov-tool gcov --add-tracefile $COVERAGE_DIR/initialize.info --add-tracefile $COVERAGE_DIR/covered.info --output-file $COVERAGE_DIR/final.info
lcov --gcov-tool gcov --extract $COVERAGE_DIR/final.info \*$SRC_DIR/\* --extract $COVERAGE_DIR/final.info \*$INCLUDE_DIR/\* --output-file $COVERAGE_DIR/coverage.info
genhtml $COVERAGE_DIR/coverage.info --output-directory $COVERAGE_DIR > $COVERAGE_DIR/genhtml.out
