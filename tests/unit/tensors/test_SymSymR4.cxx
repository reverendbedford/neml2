#include <catch2/catch.hpp>

#include "tensors/SymSymR4.h"
#include "tensors/SymR2.h"

using namespace torch::indexing;

TEST_CASE("SymSymR4", "[SymSymR4]")
{
  SECTION("initialize symmetric identity")
  {
    SymSymR4 result = SymSymR4::init(SymSymR4::FillMethod::identity_sym);
    SymSymR4 correct(torch::tensor({{{1, 0, 0, 0, 0, 0},
                                     {0, 1, 0, 0, 0, 0},
                                     {0, 0, 1, 0, 0, 0},
                                     {0, 0, 0, 1, 0, 0},
                                     {0, 0, 0, 0, 1, 0},
                                     {0, 0, 0, 0, 0, 1}}},
                                   TorchDefaults));
    REQUIRE(torch::allclose(result, correct));
    // An additional sanity check: the symmetric identity tensor is a projector, hence
    REQUIRE(torch::allclose(result * result, result));
  }

  SECTION("initialize volumetric identity")
  {
    SymSymR4 result = SymSymR4::init(SymSymR4::FillMethod::identity_vol);
    SymSymR4 correct(torch::tensor({{{1, 1, 1, 0, 0, 0},
                                     {1, 1, 1, 0, 0, 0},
                                     {1, 1, 1, 0, 0, 0},
                                     {0, 0, 0, 0, 0, 0},
                                     {0, 0, 0, 0, 0, 0},
                                     {0, 0, 0, 0, 0, 0}}},
                                   TorchDefaults) /
                     3);
    REQUIRE(torch::allclose(result, correct));
    // An additional sanity check: the volumetric identity tensor is a projector, hence
    REQUIRE(torch::allclose(result * result, result));
    // Yet another sanity check: the volumetric identity tensor should project a second order tensor
    // onto its volumetric part, hence
    SymR2 A = SymR2::init(Scalar(3), Scalar(2), Scalar(1), Scalar(5), Scalar(6), Scalar(7));
    REQUIRE(torch::allclose(result * A, A.vol()));
  }

  SECTION("initialize deviatoric identity")
  {
    SymSymR4 result = SymSymR4::init(SymSymR4::FillMethod::identity_dev);
    SymSymR4 correct(torch::tensor({{{2. / 3., -1. / 3., -1. / 3., 0., 0., 0.},
                                     {-1. / 3., 2. / 3., -1. / 3., 0., 0., 0.},
                                     {-1. / 3., -1. / 3., 2. / 3., 0., 0., 0.},
                                     {0., 0., 0., 1., 0., 0.},
                                     {0., 0., 0., 0., 1., 0.},
                                     {0., 0., 0., 0., 0., 1.}}},
                                   TorchDefaults));
    REQUIRE(torch::allclose(result, correct));
    // An additional sanity check: the deviatoric identity tensor is a projector, hence
    REQUIRE(torch::allclose(result * result, result));
    // Yet another sanity check: the deviatoric identity tensor should project a second order tensor
    // onto its deviatoric part, hence
    SymR2 A = SymR2::init(Scalar(3), Scalar(2), Scalar(1), Scalar(5), Scalar(6), Scalar(7));
    REQUIRE(torch::allclose(result * A, A.dev()));
  }

  SECTION("elasticity tensor from E and nu")
  {
    SECTION("unbatched")
    {
      Scalar E(100);
      Scalar nu(0.3);
      SymSymR4 result = SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {E, nu});
      SymSymR4 correct(torch::tensor({{{134.6154, 57.6923, 57.6923, 0.0000, 0.0000, 0.0000},
                                       {57.6923, 134.6154, 57.6923, 0.0000, 0.0000, 0.0000},
                                       {57.6923, 57.6923, 134.6154, 0.0000, 0.0000, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 76.9231, 0.0000, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 0.0000, 76.9231, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 76.9231}}},
                                     TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
    SECTION("batched")
    {
      Scalar E(torch::tensor({{100}, {200}}, TorchDefaults));
      Scalar nu(torch::tensor({{0.3}, {0.25}}, TorchDefaults));
      SymSymR4 result = SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {E, nu});
      SymSymR4 correct(torch::tensor({{{134.6154, 57.6923, 57.6923, 0.0000, 0.0000, 0.0000},
                                       {57.6923, 134.6154, 57.6923, 0.0000, 0.0000, 0.0000},
                                       {57.6923, 57.6923, 134.6154, 0.0000, 0.0000, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 76.9231, 0.0000, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 0.0000, 76.9231, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 76.9231}},
                                      {{240., 80., 80., 0.0000, 0.0000, 0.0000},
                                       {80., 240., 80., 0.0000, 0.0000, 0.0000},
                                       {80., 80., 240., 0.0000, 0.0000, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 160., 0.0000, 0.0000},
                                       {0.0000, 0.0000, 0.0000, 0.0000, 160., 0.0000},
                                       {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 160.}}},
                                     TorchDefaults));
      REQUIRE(torch::allclose(result, correct));
    }
  }

  SECTION("ijkl,kl->ij")
  {
    SymSymR4 C_unbatched =
        SymSymR4::init(SymSymR4::FillMethod::isotropic_E_nu, {Scalar(100), Scalar(0.25)});
    SymSymR4 I_unbatched = SymSymR4::init(SymSymR4::FillMethod::identity_sym);
    SymSymR4 C_batched = C_unbatched.expand_batch(2);
    SymSymR4 I_batched = I_unbatched.expand_batch(2);
    SECTION("unbatched,unbatched")
    {
      SymSymR4 result = C_unbatched * I_unbatched;
      REQUIRE(torch::allclose(result, C_unbatched));
    }
    SECTION("unbatched,batched")
    {
      SymSymR4 result = C_unbatched * I_batched;
      REQUIRE(torch::allclose(result, C_batched));
    }
    SECTION("batched,unbatched")
    {
      SymSymR4 result = C_batched * I_unbatched;
      REQUIRE(torch::allclose(result, C_batched));
    }
    SECTION("batched,batched")
    {
      SymSymR4 result = C_batched * I_batched;
      REQUIRE(torch::allclose(result, C_batched));
    }
  }
}
