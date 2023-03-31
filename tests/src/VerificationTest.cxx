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

#include "VerificationTest.h"

#include "StructuralDriver.h"

#include "neml2/misc/types.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace neml2;
using namespace torch::indexing;

VerificationTest::VerificationTest(std::string fname)
  : _filename(fname)
{
  parse();
}

bool
VerificationTest::compare(neml2::Model & model) const
{
  // Temperature needs to be added
  auto driver = StructuralDriver(
      model, time().unsqueeze(1).unsqueeze(-1), strain().unsqueeze(1), "total_strain");
  auto [all_inputs, all_outputs] = driver.run();

  // Form stress into a big tensor
  TorchSize nsteps = all_outputs.size();
  TorchSize nbatch = all_outputs[0].tensor().sizes()[0];

  auto stress = torch::empty({nsteps, nbatch, 6}, TorchDefaults);
  for (TorchSize i = 0; i < nsteps; i++)
  {
    stress.index_put_({i}, all_outputs[i].slice("state").get<SymR2>("cauchy_stress"));
  }

  return torch::allclose(this->stress(), stress.index({Slice(), 0}), rtol(), atol());
}

void
VerificationTest::parse()
{
  std::ifstream file(_filename);
  std::string buffer;

  // Simple state machine to figure out where we are in the file
  int state = 0;

  std::vector<double> time_buffer;
  std::vector<double> strain_buffer;
  std::vector<double> stress_buffer;
  std::vector<double> temperature_buffer;

  neml2::TorchSize ntime = 0;

  while (std::getline(file, buffer))
  {
    auto split = split_string(buffer);
    // Comments only allowed in the first position I guess
    if (split[0] == "#")
      continue;

    // Header
    if (state == 0)
    {
      _neml_model_file = split[0];
      _neml_model_name = split[1];
      _neml2_model_file = split[2];
      _neml2_model_name = split[3];
      if (split[4] == "no_temperature")
        _with_temperature = false;
      else if (split[4] == "with_temperature")
        _with_temperature = true;
      else
        throw std::runtime_error("Unknown temperature flag " + split[4]);

      _rtol = std::stod(split[5]);
      _atol = std::stod(split[6]);

      state += 1;
    }
    // Description line
    else if (state == 1)
    {
      _description = buffer;
      state += 1;
    }
    // Data
    else
    {
      ntime += 1;
      time_buffer.push_back(std::stod(split[0]));
      for (size_t i = 1; i < 7; i++)
        strain_buffer.push_back(std::stod(split[i]));
      for (size_t i = 7; i < 13; i++)
        stress_buffer.push_back(std::stod(split[i]));
      if (_with_temperature)
        temperature_buffer.push_back(std::stod(split[13]));
    }
  }

  // Convert buffers to tensors
  _time = torch::tensor(time_buffer, TorchDefaults);
  _strain = torch::tensor(strain_buffer, TorchDefaults).view({ntime, 6});
  _stress = torch::tensor(stress_buffer, TorchDefaults).view({ntime, 6});
  _temperature = torch::tensor(temperature_buffer, TorchDefaults);
}

std::vector<std::string>
split_string(const std::string & input, const char * delimiter)
{
  std::stringstream ss(input);
  std::vector<std::string> out;
  std::string buffer;

  while (std::getline(ss, buffer, *delimiter))
  {
    out.push_back(buffer);
  }

  return out;
}
