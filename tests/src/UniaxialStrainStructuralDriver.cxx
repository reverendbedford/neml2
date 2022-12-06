#include "UniaxialStrainStructuralDriver.h"

UniaxialStrainStructuralDriver::UniaxialStrainStructuralDriver(Model & model,
                                                               Scalar max_strain,
                                                               Scalar end_time,
                                                               TorchSize nsteps)
  : _model(model),
    _max_strain(max_strain),
    _end_time(end_time),
    _nsteps(nsteps),
    _nbatch(end_time.batch_sizes()[0])
{
}

std::tuple<std::vector<LabeledVector>, std::vector<LabeledVector>>
UniaxialStrainStructuralDriver::run()
{
  // Create 2 LabeledMatrix to store the inputs and outputs
  std::vector<LabeledVector> all_inputs(_nsteps + 1);
  std::vector<LabeledVector> all_outputs(_nsteps + 1);

  // strain increment and time increment
  Scalar delta_strain_x = _max_strain / _nsteps;
  Scalar delta_strain_y = -0.5 * delta_strain_x;
  Scalar delta_strain_z = -0.5 * delta_strain_x;
  SymR2 delta_strain = SymR2::init(delta_strain_x, delta_strain_y, delta_strain_z);
  Scalar dt = _end_time / _nsteps;

  // Initialize
  auto in = LabeledVector(_nbatch, _model.input());
  auto out = LabeledVector(_nbatch, _model.output());

  // Initialize the old state it if necessary
  // For example _model.init_state(in);

  // Initialize the old forces
  Scalar current_time = Scalar(0, _nbatch);
  SymR2 current_strain = SymR2::zeros().expand_batch(_nbatch);
  in.slice(0, "old_forces").set(current_time, "time");
  in.slice(0, "old_forces").set(current_strain, "total_strain");

  for (TorchSize i = 0; i < _nsteps + 1; i++)
  {
    // Advance the step
    current_time = current_time + dt;
    current_strain = current_strain + delta_strain;
    in.slice(0, "forces").set(current_strain, "total_strain");
    in.slice(0, "forces").set(current_time, "time");

    // Perform the constitutive update
    out = solve_step(in, i, all_inputs, all_outputs);

    // Propagate the forces and state in time
    // current --> old
    in.slice(0, "old_state").copy(out.slice(0, "state"));
    in.slice(0, "old_forces").copy(in.slice(0, "forces"));
  }

  return {all_inputs, all_outputs};
}

LabeledVector
UniaxialStrainStructuralDriver::solve_step(LabeledVector in,
                                           TorchSize i,
                                           std::vector<LabeledVector> & all_inputs,
                                           std::vector<LabeledVector> & all_outputs) const
{
  auto out = _model.value(in);

  // Store the results
  all_inputs[i] = in.clone();
  all_outputs[i] = out.clone();

  return out;
}
