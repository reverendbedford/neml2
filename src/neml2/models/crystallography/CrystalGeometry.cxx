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

#include "neml2/models/crystallography/CrystalGeometry.h"

#include "neml2/models/crystallography/CrystalClass.h"
#include "neml2/models/crystallography/MillerIndex.h"
#include "neml2/tensors/tensors.h"

using namespace torch::indexing;

namespace neml2
{
namespace crystallography
{

register_NEML2_object(CrystalGeometry);

OptionSet
CrystalGeometry::expected_options()
{
  OptionSet options = StaticModel::expected_options();
  options.set<std::string>("crystal_class");
  options.set<CrossRef<Vec>>("lattice_vectors");
  options.set<CrossRef<MillerIndex>>("slip_directions");
  options.set<CrossRef<MillerIndex>>("slip_planes");

  return options;
}

CrystalGeometry::CrystalGeometry(const OptionSet & options)
  : CrystalGeometry(options,
                    setup_schmid_tensors(Factory::get_object<CrystalClass>(
                                             "Models", options.get<std::string>("crystal_class")),
                                         options.get<CrossRef<MillerIndex>>("slip_directions"),
                                         options.get<CrossRef<MillerIndex>>("slip_planes")))
{
}

Vec
CrystalGeometry::a1() const
{
  return _lattice_vectors.batch_index({0});
}

Vec
CrystalGeometry::a2() const
{
  return _lattice_vectors.batch_index({1});
}

Vec
CrystalGeometry::a3() const
{
  return _lattice_vectors.batch_index({2});
}

Vec
CrystalGeometry::b1() const
{
  return _reciprocal_lattice_vectors.batch_index({0});
}

Vec
CrystalGeometry::b2() const
{
  return _reciprocal_lattice_vectors.batch_index({1});
}

Vec
CrystalGeometry::b3() const
{
  return _reciprocal_lattice_vectors.batch_index({2});
}

CrystalGeometry::CrystalGeometry(const OptionSet & options,
                                 std::pair<R2, std::vector<TorchSize>> slip_res)
  : StaticModel(options),
    _class(include_model<CrystalClass>(options.get<std::string>("crystal_class"))),
    _lattice_vectors(declare_buffer<Vec>("lattice_vectors", "lattice_vectors")),
    _reciprocal_lattice_vectors(declare_buffer<Vec>("reciprocal_lattice_vectors",
                                                    make_reciprocal_lattice(_lattice_vectors))),
    _slip_directions(declare_buffer<MillerIndex>("slip_directions", "slip_directions")),
    _slip_planes(declare_buffer<MillerIndex>("slip_planes", "slip_planes")),
    _A(declare_buffer<R2>("schmid_tensors", slip_res.first)),
    _slip_offsets(slip_res.second),
    _M(declare_buffer<SR2>("symmetric_schmid_tensors", SR2(_A))),
    _W(declare_buffer<WR2>("skew_symmetric_schmid_tensors", WR2(_A)))
{
}

Vec
CrystalGeometry::make_reciprocal_lattice(const Vec & lattice_vectors)
{
  auto a1 = lattice_vectors.batch_index({0});
  auto a2 = lattice_vectors.batch_index({1});
  auto a3 = lattice_vectors.batch_index({2});

  Vec rl = Vec(torch::stack({a2.cross(a3) / a1.dot(a2.cross(a3)),
                             a3.cross(a1) / a2.dot(a3.cross(a1)),
                             a1.cross(a2) / a3.dot(a1.cross(a2))}));

  return rl;
}

std::pair<R2, std::vector<TorchSize>>
CrystalGeometry::setup_schmid_tensors(const CrystalClass & cls,
                                      const MillerIndex & slip_directions,
                                      const MillerIndex & slip_planes)
{
  if (slip_directions.batch_sizes() != slip_planes.batch_sizes())
    neml_assert("Input slip directions and planes must have the same batch sizes");

  auto bshape = slip_planes.batch_sizes();
  auto nbatch = slip_planes.batch_dim();

  // Loop over each slip system
  std::vector<R2> cart_slip_directions;
  std::vector<R2> cart_slip_planes;
  std::vector<TorchSize> offsets = {0};
  for (TorchSize i = 0; i < bshape[nbatch - 1]; i++)
  {
    auto cmd = slip_directions.batch_index({torch::indexing::Ellipsis, i});
    auto cmp = slip_planes.batch_index({torch::indexing::Ellipsis, i});

    std::cout << cls.unique_bidirectional(cmd.to_vec()) << std::endl;
    std::cout << cls.unique_bidirectional(cmp.to_vec()) << std::endl;
  }

  return std::make_pair(R2::fill(1.0), offsets);
}

} // namespace crystallography
} // namespace neml2