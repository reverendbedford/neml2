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

#pragma once

#include "neml2/drivers/Driver.h"
#include <filesystem>

namespace neml2
{
/**
 * @brief The driver for a transient initial-value problem.
 *
 */
class TransientDriver : public Driver
{
public:
  static ParameterSet expected_params();

  /**
   * @brief Construct a new TransientDriver object
   *
   * @param params The parameters extracted from the input file
   */
  TransientDriver(const ParameterSet & params);

  bool run() override;

  /// The destination file/path to save the results.
  virtual std::string save_as_path() const;

  /**
   * @brief The results (input and output) from all time steps.
   *
   * @return torch::nn::ModuleDict The results (input and output) from all time steps. Keys of the
   * dict are "input" and "output". Each buffer in the submodules correspond to a variable.
   */
  virtual torch::nn::ModuleDict result() const;

protected:
  virtual void check_integrity() const override;

  /// Solve the initial value problem
  virtual bool solve();

  // @{ Routines that are called every step
  /// Advance in time: the state becomes old state, and forces become old forces.
  virtual void advance_step();
  /// Update the driving forces for the current time step.
  virtual void update_forces();
  /// Apply the initial conditions.
  virtual void apply_ic();
  /// Apply the predictor to calculate the initial guess for the current time step.
  virtual void apply_predictor();
  /// Perform the constitutive update for the current time step.
  virtual void solve_step();
  /// Save the results of the current time step.
  virtual void store_step();
  // @}

  /// Save the results into the destination file/path.
  virtual void output() const;

  /// The model which the driver uses to perform constitutive updates.
  Model & _model;
  /// The device on which all the computation occurs
  const torch::Device _device;

  /// The current time
  torch::Tensor _time;
  /// The current step count
  TorchSize _step_count;
  /// LabeledAxisAccessor for the time
  LabeledAxisAccessor _time_name;
  /// Total number of steps
  TorchSize _nsteps;
  /// The batch size
  TorchSize _nbatch;
  /// The input to the constitutive model
  LabeledVector _in;
  /// The output of the constitutive model
  LabeledVector _out;

  /// The predictor used to set the initial guess
  std::string _predictor;
  /// The destination file name or file path
  std::string _save_as;
  /// Set to true to list all the model parameters at the beginning.
  const bool _show_params;

  /// Inputs from all time steps
  LabeledTensor<2, 1> _result_in;
  /// Outputs from all time steps
  LabeledTensor<2, 1> _result_out;

private:
  void output_pt(const std::filesystem::path & out) const;
};
} // namespace neml2
