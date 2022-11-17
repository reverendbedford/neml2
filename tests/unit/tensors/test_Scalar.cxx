#include <catch2/catch.hpp>

#include "tensors/Scalar.h"

TEST_CASE("Unbatched Scalar", "[Scalar]")
{
  SECTION("construct from plain data type")
  {
    Scalar a = 2.5;
    torch::Tensor correct(torch::tensor({2.5}, TorchDefaults));
    REQUIRE(torch::allclose(a, correct));
  }

  SECTION("+ unbatched Scalar")
  {
    Scalar a = 2.5;
    Scalar b = 3.1;
    Scalar result = a + b;
    Scalar correct = 5.6;
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("+ batched Scalar")
  {
    int nbatch = 5;
    Scalar a = 2.5;
    Scalar b(3.1, nbatch);
    Scalar result = a + b;
    Scalar correct(5.6, nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("- unbatched Scalar")
  {
    Scalar a = 2.5;
    Scalar b = 3.1;
    Scalar result = a - b;
    Scalar correct = -0.6;
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("- batched Scalar")
  {
    int nbatch = 5;
    Scalar a = 2.5;
    Scalar b(3.1, nbatch);
    Scalar result = a - b;
    Scalar correct(-0.6, nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("* unbatched Scalar")
  {
    Scalar a = 2.5;
    Scalar b = 3.1;
    Scalar result = a * b;
    Scalar correct = 2.5 * 3.1;
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("* batched Scalar")
  {
    int nbatch = 5;
    Scalar a = 2.5;
    Scalar b(3.1, nbatch);
    Scalar result = a * b;
    Scalar correct(2.5 * 3.1, nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("/ unbatched Scalar")
  {
    Scalar a = 2.5;
    Scalar b = 3.1;
    Scalar result = a / b;
    Scalar correct = 2.5 / 3.1;
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("/ batched Scalar")
  {
    int nbatch = 5;
    Scalar a = 2.5;
    Scalar b(3.1, nbatch);
    Scalar result = a / b;
    Scalar correct(2.5 / 3.1, nbatch);
    REQUIRE(torch::allclose(result, correct));
  }
}

TEST_CASE("Batched Scalar", "[Scalar]")
{
  int nbatch = 5;

  SECTION("construct from plain data type")
  {
    Scalar a(2.5, nbatch);
    torch::Tensor correct(torch::tensor({2.5, 2.5, 2.5, 2.5, 2.5}, TorchDefaults));
    REQUIRE(torch::allclose(a, correct));
  }

  SECTION("+ unbatched Scalar")
  {
    Scalar a(2.5, nbatch);
    Scalar b = 3.1;
    Scalar result = a + b;
    Scalar correct(5.6, nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("+ batched Scalar")
  {
    Scalar a(2.5, nbatch);
    Scalar b(3.1, nbatch);
    Scalar result = a + b;
    Scalar correct(5.6, nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("- unbatched Scalar")
  {
    Scalar a(2.5, nbatch);
    Scalar b = 3.1;
    Scalar result = a - b;
    Scalar correct(-0.6, nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("- batched Scalar")
  {
    Scalar a(2.5, nbatch);
    Scalar b(3.1, nbatch);
    Scalar result = a - b;
    Scalar correct(-0.6, nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("* unbatched Scalar")
  {
    Scalar a(2.5, nbatch);
    Scalar b = 3.1;
    Scalar result = a * b;
    Scalar correct(2.5 * 3.1, nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("* batched Scalar")
  {
    Scalar a(2.5, nbatch);
    Scalar b(3.1, nbatch);
    Scalar result = a * b;
    Scalar correct(2.5 * 3.1, nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("/ unbatched Scalar")
  {
    Scalar a(2.5, nbatch);
    Scalar b = 3.1;
    Scalar result = a / b;
    Scalar correct(2.5 / 3.1, nbatch);
    REQUIRE(torch::allclose(result, correct));
  }

  SECTION("/ batched Scalar")
  {
    Scalar a(2.5, nbatch);
    Scalar b(3.1, nbatch);
    Scalar result = a / b;
    Scalar correct(2.5 / 3.1, nbatch);
    REQUIRE(torch::allclose(result, correct));
  }
}

TEST_CASE("Scalar can't be created with semantically non-scalar tensors", "[Scalar]")
{
  // This can't happen as the tensor dimension is not (1,)
  REQUIRE_THROWS(Scalar(torch::zeros({2, 2}, TorchDefaults)));
}
