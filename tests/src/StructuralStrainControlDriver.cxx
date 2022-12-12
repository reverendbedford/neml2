#include "StructuralStrainControlDriver.h"

using namespace neml2;

StructuralStrainControlDriver::StructuralStrainControlDriver(const neml2::Model & model,
                                                             torch::Tensor time,
                                                             torch::Tensor strain)
  : _model(model),
    _time(time),
    _strain(strain),
    _nsteps(strain.sizes()[0]),
    _nbatch(strain.sizes()[1])
{
  neml_assert(time.dim() == 3,
              "time should have dimension 3 "
              "but instead has dimension ",
              time.dim());
  neml_assert(strain.dim() == 3,
              "strain should have dimension 3 "
              "but instead has dimension",
              strain.dim());
  neml_assert(time.sizes()[0] == strain.sizes()[0],
              "strain and time should have the "
              "same number of time steps. "
              "The input time has ",
              time.sizes()[0],
              " time steps, "
              "while the input strain has ",
              strain.sizes()[0],
              " time steps");
  neml_assert(time.sizes()[1] == strain.sizes()[1],
              "strain and time should have the "
              "same batch size.  The input time has a batch size of ",
              time.sizes()[1],
              " while the input strain has a batch "
              "size of ",
              strain.sizes()[1]);
  neml_assert(time.sizes()[2] == 1,
              "Input time should have final "
              "dimension 1 but instead has final dimension ",
              time.sizes()[2]);
  neml_assert(strain.sizes()[2] == 6,
              "Input strain should have final "
              "dimension 6 but instead has final dimension ",
              strain.sizes()[2]);
}

std::tuple<std::vector<LabeledVector>, std::vector<LabeledVector>>
StructuralStrainControlDriver::run()
{
  // Create 2 LabeledMatrix to store the inputs and outputs
  std::vector<LabeledVector> all_inputs(_nsteps);
  std::vector<LabeledVector> all_outputs(_nsteps);

  // Initialize
  auto in = LabeledVector(_nbatch, _model.input());
  auto out = LabeledVector(_nbatch, _model.output());

  // Initialize the old state it if necessary
  // For example _model.init_state(in);

  // Initialize the old forces
  Scalar current_time = Scalar(_time.index({0}));
  SymR2 current_strain = SymR2(_strain.index({0}));
  in.slice(0, "old_forces").set(current_time, "time");
  in.slice(0, "old_forces").set(current_strain, "total_strain");

  all_inputs[0] = in.clone();
  all_outputs[0] = out.clone();

  for (TorchSize i = 1; i < _nsteps; i++)
  {
    // Advance the step
    current_time = Scalar(_time.index({i}));
    current_strain = SymR2(_strain.index({i}));
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
StructuralStrainControlDriver::solve_step(LabeledVector in,
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
