#define CATCH_CONFIG_MAIN

#ifdef ENABLE_BENCHMARK
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#endif

#include <catch2/catch.hpp>

#ifdef ENABLE_BENCHMARK
#include "tensor/Scalar.h"
#include "tensor/SymR2.h"
#include "tensor/SymSymR4.h"

using namespace neml2;

class BenchmarkCommon
{
public:
  BenchmarkCommon()
    : nbatches(
          {10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000, 2560000, 5120000, 10240000})
  {
  }

  std::string bname(const std::string method_name, TorchSize nbatch, TorchSize chunk_size = 1) const
  {
    return "{" + std::to_string(nbatch) + "} {" + std::to_string(chunk_size) + "} " + method_name;
  }

  std::vector<TorchSize> nbatches;
};

TEST_CASE_METHOD(BenchmarkCommon, "Benchmark Scalar", "[BENCHMARK][Scalar]")
{
  Scalar a_unbatched = 1.23123;
  for (TorchSize nbatch : nbatches)
  {
    Scalar a_batched(1.23123, nbatch);
    Scalar b_batched(-5.3432, nbatch);
    BENCHMARK(bname("UScalar+BScalar", nbatch)) { return a_unbatched + b_batched; };
    BENCHMARK(bname("BScalar+BScalar", nbatch)) { return a_batched + b_batched; };
  }
}

TEST_CASE_METHOD(BenchmarkCommon, "Benchmark SymR2", "[BENCHMARK][SymR2]")
{
  SymR2 A_unbatched = SymR2::init(1, 2, 3, 4, 5, 6);
  for (TorchSize nbatch : nbatches)
  {
    Scalar a(1, nbatch);
    SymR2 A_batched = SymR2::init(1, 2, 3, 4, 5, 6).expand_batch(nbatch);
    SymR2 B_batched = SymR2::init(5, 6, 7, 8, 9, 10).expand_batch(nbatch);
    BENCHMARK(bname("USymR2+BSymR2", nbatch)) { return A_unbatched + B_batched; };
    BENCHMARK(bname("BSymR2+BSymR2", nbatch)) { return A_batched + B_batched; };
    BENCHMARK(bname("tr(BSymR2)", nbatch)) { return A_batched.tr(); };
    BENCHMARK(bname("vol(BSymR2)", nbatch)) { return A_batched.vol(); };
    BENCHMARK(bname("dev(BSymR2)", nbatch)) { return A_batched.dev(); };
    BENCHMARK(bname("det(BSymR2)", nbatch)) { return A_batched.det(); };
    BENCHMARK(bname("norm(BSymR2)", nbatch)) { return A_batched.norm(); };
    BENCHMARK(bname("(USymR2)_ij(BSymR2)_ij", nbatch)) { return A_unbatched.inner(B_batched); };
    BENCHMARK(bname("(BSymR2)_ij(BSymR2)_ij", nbatch)) { return A_batched.inner(B_batched); };
    BENCHMARK(bname("(USymR2)_ij(BSymR2)_kl", nbatch)) { return A_unbatched.outer(B_batched); };
    BENCHMARK(bname("(BSymR2)_ij(BSymR2)_kl", nbatch)) { return A_batched.outer(B_batched); };
  }
}

TEST_CASE_METHOD(BenchmarkCommon, "Benchmark SymSymR4", "[BENCHMARK][SymSymR4]")
{
  SymSymR4 C_unbatched = SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {100, 0.25});
  for (TorchSize nbatch : nbatches)
  {
    Scalar E = 100;
    Scalar nu = 0.25;
    SymR2 A_batched = SymR2::init(1, 2, 3, 4, 5, 6).expand_batch(nbatch);
    SymSymR4 C_batched =
        SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {100, 0.25}).expand_batch(nbatch);
    SymSymR4 D_batched =
        SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {200, 0.3}).expand_batch(nbatch);
    // clang-format off
    BENCHMARK(bname("USymSymR4+BSymSymR4", nbatch)) { return C_unbatched + D_batched; };
    BENCHMARK(bname("BSymSymR4+BSymSymR4", nbatch)) { return C_batched + D_batched; };
    BENCHMARK(bname("(USymSymR4)_ijkl(BSymSymR4)_klmn", nbatch)) { return C_unbatched * D_batched; };
    BENCHMARK(bname("(BSymSymR4)_ijkl(BSymSymR4)_klmn", nbatch)) { return C_batched * D_batched; };
    BENCHMARK(bname("(USymSymR4)_ijkl(BSymR2)_kl", nbatch)) { return C_unbatched * A_batched; };
    BENCHMARK(bname("(BSymSymR4)_ijkl(BSymR2)_kl", nbatch)) { return C_batched * A_batched; };
    // clang-format on
  }
}
#endif // ENABLE_BENCHMARK
