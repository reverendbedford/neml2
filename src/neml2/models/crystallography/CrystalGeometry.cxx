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
  : StaticModel(options),
    _class(include_model<CrystalClass>(options.get<std::string>("crystal_class"))),
    _lattice_vectors(declare_buffer<Vec>("lattice_vectors", "lattice_vectors")),
    _slip_directions(declare_buffer<MillerIndex>("slip_directions", "slip_directions")),
    _slip_planes(declare_buffer<MillerIndex>("slip_planes", "slip_planes")),
    _A(declare_buffer<R2>("schmid_tensors", setup_schmid_tensors(_slip_directions, _slip_planes))),
    _M(declare_buffer<SR2>("symmetric_schmid_tensors", SR2(_A))),
    _W(declare_buffer<WR2>("skew_symmetric_schmid_tensors", WR2(_A)))
{
}

R2
CrystalGeometry::setup_schmid_tensors(const MillerIndex & slip_directions,
                                      const MillerIndex & slip_planes)
{
  if (slip_directions.batch_sizes() != slip_planes.batch_sizes())
    neml_assert("Input slip directions and planes must have the same batch sizes");
  return R2::fill(1.0);
}

} // namespace crystallography
} // namespace neml2