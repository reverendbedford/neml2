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

#include "neml2/models/crystallography/CrystalClass.h"

using namespace torch::indexing;

namespace neml2
{
namespace crystallography
{

SymmetryOperator
symmetry_operations_from_orbifold(std::string orbifold, const torch::TensorOptions & options)
{
  if (orbifold == "432")
  {
    return SymmetryOperator::from_quaternion(Quaternion(crystal_symmetry_operators::cubic))
        .to(options);
  }
  else if (orbifold == "23")
  {
    return SymmetryOperator::from_quaternion(
               Quaternion(crystal_symmetry_operators::cubic.index({Slice(0, 12)})))
        .to(options);
  }
  else if (orbifold == "622")
  {
    return SymmetryOperator::from_quaternion(Quaternion(crystal_symmetry_operators::hexagonal))
        .to(options);
  }
  else if (orbifold == "32")
  {
    return SymmetryOperator::from_quaternion(
               Quaternion(
                   torch::cat({crystal_symmetry_operators::hexagonal.index({Slice(0, 3)}),
                               crystal_symmetry_operators::hexagonal.index({Slice(9, 12)})})))
        .to(options);
  }
  else if (orbifold == "6")
  {
    return SymmetryOperator::from_quaternion(
               Quaternion(crystal_symmetry_operators::hexagonal.index({Slice(0, 6)})))
        .to(options);
  }
  else if (orbifold == "3")
  {
    return SymmetryOperator::from_quaternion(
               Quaternion(crystal_symmetry_operators::hexagonal.index({Slice(0, 3)})))
        .to(options);
  }
  else if (orbifold == "42")
  {
    return SymmetryOperator::from_quaternion(Quaternion(crystal_symmetry_operators::tetragonal))
        .to(options);
  }
  else if (orbifold == "4")
  {
    return SymmetryOperator::from_quaternion(
               Quaternion(
                   torch::cat({crystal_symmetry_operators::tetragonal.index({Slice(0, 1)}),
                               crystal_symmetry_operators::tetragonal.index({Slice(3, 6)})})))
        .to(options);
  }
  else if (orbifold == "222")
  {
    return SymmetryOperator::from_quaternion(
               Quaternion(crystal_symmetry_operators::tetragonal.index({Slice(0, 4)})))
        .to(options);
  }
  else if (orbifold == "2")
  {
    return SymmetryOperator::from_quaternion(
               Quaternion(crystal_symmetry_operators::tetragonal.index({Slice(0, 2)})))
        .to(options);
  }
  else if (orbifold == "1")
  {
    return SymmetryOperator::from_quaternion(
               Quaternion(crystal_symmetry_operators::tetragonal.index({Slice(0, 1)})))
        .to(options);
  }
  else
  {
    throw NEMLException("Unknown crystal class " + orbifold);
  }
}

register_NEML2_object(CrystalClass);

OptionSet
CrystalClass::expected_options()
{
  OptionSet options = Model::expected_options();
  options.set<std::string>("orbifold");
  return options;
}

CrystalClass::CrystalClass(const OptionSet & options)
  : Model(options),
    _operations(declare_buffer<SymmetryOperator>(
        "operations", symmetry_operations_from_orbifold(options.get<std::string>("orbifold"))))
{
}

void
CrystalClass::set_value(const LabeledVector & in,
                        LabeledVector * out,
                        LabeledMatrix * dout_din,
                        LabeledTensor3D * d2out_din2) const
{
  (void)in;
  (void)out;
  (void)dout_din;
  (void)d2out_din2;
  return;
}

const SymmetryOperator &
CrystalClass::operations() const
{
  return _operations;
}

TorchSize
CrystalClass::size() const
{
  return _operations.batch_sizes()[0];
}

}
} // namespace neml2