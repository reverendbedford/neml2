#include "VerificationTest.h"

#include "misc/types.h"

#include <iostream>
#include <fstream>
#include <sstream>

VerificationTest::VerificationTest(std::string fname) :
    _filename(fname)
{
  parse();
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

  while (std::getline(file, buffer)) {
    auto split = split_string(buffer);
    // Comments only allowed in the first position I guess
    if (split[0] == "#") continue;
    
    // Header
    if (state == 0) {
      _neml_model_file = split[0];
      _neml_model_name = split[1];
      _neml2_model_file = split[2];
      _neml2_model_name = split[3];
      if (split[4] == "no_temperature") _with_temperature = false;
      else if (split[4] == "with_temperature") _with_temperature = true;
      else throw std::runtime_error("Unknown temperature flag " + 
                                    split[4]);
      state += 1;
    }
    // Description line
    else if (state == 1) {
      _description = buffer;
      state += 1;
    }
    // Data
    else {
      ntime += 1;
      time_buffer.push_back(std::stod(split[0]));
      for (size_t i = 1; i < 7; i++) strain_buffer.push_back(std::stod(split[i]));
      for (size_t i = 7; i < 13; i++) stress_buffer.push_back(std::stod(split[i]));
      if (_with_temperature)
        temperature_buffer.push_back(std::stod(split[13]));
    }
  }

  // Convert buffers to tensors
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

  while (std::getline(ss, buffer, *delimiter)) {
    out.push_back(buffer);
  }

  return out;
}
