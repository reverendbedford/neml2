#include "UniaxialStrainStructuralDriver.h"
#include "models/ConstitutiveModel.h"

UniaxialStrainStructuralDriver::UniaxialStrainStructuralDriver(ConstitutiveModel & model,
                                                               torch::Tensor max_strain,
                                                               torch::Tensor strain_rate)
  : _model(model),
    _max_strain(max_strain),
    _strain_rate(strain_rate)
{
  // Check that the shapes were correct
  if ((_max_strain.sizes().size() != 1) || (_strain_rate.sizes().size() != 1))
    throw std::runtime_error("max_strain and strain_rate must be 1D");

  if (_max_strain.sizes()[0] != _strain_rate.sizes()[0])
    throw std::runtime_error("max_strain and strain_rate must have the "
                             "same size, equal to the batch size");
}

TorchSize
UniaxialStrainStructuralDriver::batch_size() const
{
  return _max_strain.sizes()[0];
}

void
UniaxialStrainStructuralDriver::run(TorchSize nsteps)
{
  // Setup storage for the forces and states arrays
  TorchShape forces_shape = _model.forces().required_shape(batch_size());
  forces_shape.insert(forces_shape.begin(), nsteps);
  _forces = torch::zeros(forces_shape, TorchDefaults);
  TorchShape states_shape = _model.state().required_shape(batch_size());
  states_shape.insert(states_shape.begin(), nsteps);
  _states = torch::zeros(states_shape, TorchDefaults);

  // Figure out where to insert the uniaxial strain and time
  TorchSize strain_index =
      _model.forces().item_offsets()[_model.forces().item_locations().at("strain")];
  TorchSize time_index =
      _model.forces().item_offsets()[_model.forces().item_locations().at("time")];

  // Setup time and strain in the forces array
  for (TorchSize i = 0; i < batch_size(); i++)
  {
    // 1st entry in strain: linspace between 0 and max_strain
    _forces.index_put_({Slice(), i, strain_index},
                       torch::linspace(0, _max_strain[i].item<Real>(), nsteps, TorchDefaults));
    // 2nd and 3rd entries: -0.5 * first
    _forces.index_put_({Slice(), i, strain_index + 1},
                       torch::linspace(0, _max_strain[i].item<Real>(), nsteps, TorchDefaults) *
                           -0.5);
    _forces.index_put_({Slice(), i, strain_index + 2},
                       torch::linspace(0, _max_strain[i].item<Real>(), nsteps, TorchDefaults) *
                           -0.5);

    // time: strain / strain_rate
    _forces.index_put_(
        {Slice(), i, time_index},
        torch::linspace(
            0, _max_strain[i].item<Real>() / _strain_rate[i].item<Real>(), nsteps, TorchDefaults));
  }

  // Setup initial state
  State state_n(_model.state(), _states.index({0}));
  _model.initial_state(state_n);

  // Loop through time to run the model
  for (TorchSize i = 1; i < nsteps; i++)
  {
    State forces_n(_model.forces(), _forces.index({i - 1}));
    State forces_np1(_model.forces(), _forces.index({i}));
    State state_n(_model.state(), _states.index({i - 1}));

    State state_np1 = _model.state_update(forces_np1, state_n, forces_n);

    _states.index_put_({i}, state_np1.tensor());
  }
}

void
write_csv(torch::Tensor tensor, std::string fname, std::string delimiter)
{
  std::ofstream buffer(fname);

  // First line is shape
  for (size_t i = 0; i < tensor.sizes().size(); i++)
  {
    buffer << tensor.sizes()[i];
    if (i != tensor.sizes().size() - 1)
      buffer << delimiter;
  }
  buffer << std::endl;

  // Flatten and write array
  tensor = tensor.flatten().contiguous().cpu();

  Real * ptr = tensor.data_ptr<Real>();

  for (TorchSize i = 0; i < tensor.sizes()[0]; i++)
  {
    buffer << *ptr++;
    if (i != tensor.sizes()[0] - 1)
      buffer << delimiter;
  }
}
