mkdir -p Testing/Temporary
./tests/tests $1 --benchmark-samples $2 --use-colour no -o Testing/Temporary/benchmark.out
python scripts/extract_timings.py Testing/Temporary/benchmark.out Testing/Temporary/benchmark.csv
python scripts/analyze_timings.py Testing/Temporary/benchmark.csv benchmark
