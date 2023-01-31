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

#include <catch2/catch.hpp>

#include <fstream>

#include "StructuralDriver.h"
#include "neml2/misc/math.h"
#include "neml2/base/HITParser.h"

using namespace neml2;

TEST_CASE("Chaboche regression", "[Chaboche]")
{
  HITParser parser;
  parser.parse_and_manufacture("regression/models/solid_mechanics/regression_chaboche.i");
  auto & model = Factory::get_object<Model>("Models", "model");

  TorchSize nbatch = 20;
  TorchSize nsteps = 100;
  Real max_strain = 0.10;
  Real min_time = -1;
  Real max_time = 5;

  Scalar end_time = torch::logspace(min_time, max_time, nbatch, 10, TorchDefaults).unsqueeze(-1);
  SymR2 end_strain =
      SymR2::init(max_strain, -0.5 * max_strain, -0.5 * max_strain).batch_expand(nbatch);

  BatchTensor<1> times = math::linspace<1>(torch::zeros_like(end_time), end_time, nsteps);
  BatchTensor<1> strains = math::linspace<1>(torch::zeros_like(end_strain), end_strain, nsteps);

  StructuralDriver driver(model, times, strains, "total_strain");
  torch::NoGradGuard no_grad_guard;
  auto [all_inputs, all_outputs] = driver.run();

  std::ofstream ofile;
  std::string fname = "regression/models/solid_mechanics/chaboche";

  // I use this to write csv for visualization purposes.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // for (TorchSize batch = 0; batch < nbatch; batch++)
  // {
  //   ofile.open(fname + "_forces_batch_" + utils::stringify(batch) + ".csv");
  //   for (size_t i = 0; i < all_inputs.size(); i++)
  //     LabeledVector(all_inputs[i].slice("forces")).write(ofile, ",", batch, i == 0);
  //   ofile.close();

  //   ofile.open(fname + "_state_batch_" + utils::stringify(batch) + ".csv");
  //   for (size_t i = 0; i < all_outputs.size(); i++)
  //     LabeledVector(all_outputs[i].slice("state")).write(ofile, ",", batch, i == 0);
  //   ofile.close();
  // }
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  auto inputs =
      torch::zeros(utils::add_shapes(TorchShape({nsteps}), all_inputs[0].tensor().sizes()));
  auto outputs =
      torch::zeros(utils::add_shapes(TorchShape({nsteps}), all_outputs[0].tensor().sizes()));

  for (TorchSize i = 0; i < nsteps; i++)
  {
    inputs.index_put_({i, torch::indexing::Ellipsis}, all_inputs[i].tensor());
    outputs.index_put_({i, torch::indexing::Ellipsis}, all_outputs[i].tensor());
  }

  // Below is what I used to save the results
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // torch::save(inputs, fname + "_inputs.pt");
  // torch::save(outputs, fname + "_outputs.pt");
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Load it back
  auto inputs_ref =
      torch::zeros(utils::add_shapes(TorchShape({nsteps}), all_inputs[0].tensor().sizes()));
  auto outputs_ref =
      torch::zeros(utils::add_shapes(TorchShape({nsteps}), all_outputs[0].tensor().sizes()));
  torch::load(inputs_ref, fname + "_inputs.pt");
  torch::load(outputs_ref, fname + "_outputs.pt");

  REQUIRE(torch::allclose(inputs, inputs_ref));
  REQUIRE(torch::allclose(outputs, outputs_ref));
}
