#include "models/solid_mechanics/InelasticModel.h"
#include "models/solid_mechanics/SmallStrainMechanicalModel.h"

InelasticModel::InelasticModel(SymSymR4 C,
                               InelasticFlowRate & rate,
                               InelasticFlowDirection & direction,
                               InelasticHardening & hardening)
  : _C(C),
    _rate(rate),
    _direction(direction),
    _hardening(hardening)
{
}

StateInfo
InelasticModel::state() const
{
  StateInfo si;
  si.add_substate("stress_interface", SmallStrainMechanicalModel::state());
  si.add_substate("hardening_interface", _hardening.state());
  return si;
}

StateInfo
InelasticModel::output() const
{
  return state().add_suffix("_rate");
}

StateInfo
InelasticModel::forces() const
{
  return SmallStrainMechanicalModel::forces();
}

void
InelasticModel::initial_state(State & state) const
{
  SmallStrainMechanicalModel::initial_state(state);
  _hardening.initial_state(state);
}

State
InelasticModel::value(StateInput input)
{
  State curr_state = input[0];
  State curr_force = input[1];
  State force_rate = input[2];
  State state_rate = State(output(), curr_state.batch_size());

  SymR2 inelastic_strain_rate = _rate.value(curr_state).get<Scalar>("flow_rate") *
                                _direction.value(curr_state).get<SymR2>("flow_direction");
  state_rate.set<SymR2>("stress_rate",
                        _C * (force_rate.get<SymR2>("strain_rate") - inelastic_strain_rate));
  state_rate.set_substate("hardening_interface_rate", _hardening.value(curr_state));

  return state_rate;
}

StateDerivativeOutput
InelasticModel::dvalue(StateInput input)
{
  State curr_state = input[0];
  State curr_force = input[1];
  State force_rate = input[2];
  StateDerivative dstate(output(), state(), curr_state.batch_size());
  StateDerivative dforce(output(), SmallStrainMechanicalModel::forces(), curr_state.batch_size());
  StateDerivative dforcerate(
      RateForm::output(),
      SmallStrainMechanicalModel::forces().remove("time").add_suffix("_rate"),
      curr_state.batch_size());

  // TODO improve the flow of all of this
  // dstate is dense (in the block sense)
  // d(plastic_strain_rate)/d(state)
  State rate = _rate.value(curr_state);
  State dir = _direction.value(curr_state);
  StateDerivative A = _direction.dvalue(curr_state).scalar_product(rate.get<Scalar>("flow_rate"));
  StateDerivative B = dir.promote_right("flow_rate").chain(_rate.dvalue(curr_state));
  StateDerivative dep_dstate = A + B;
  // d(stress_rate)/d(state)
  StateDerivative ds_dstate = StateDerivative::promote("stress_rate",
                                                       "flow_direction",
                                                       -_C.expand_batch(curr_state.batch_size()))
                                  .chain(dep_dstate);

  // d(hardening_interface_rate)/d(state) comes directly from the
  // InelasticHardening object
  StateDerivative dh_dstate = _hardening.dvalue(curr_state);

  // Insert in the right slots
  dstate.set_slice("stress_interface_rate", ds_dstate);
  dstate.set_slice("hardening_interface_rate", dh_dstate);

  // dforce is zero for now

  // dforcerate has a single non-zero block
  dforcerate.set<SymSymR4>("stress_rate", "strain_rate", _C.expand_batch(curr_state.batch_size()));

  return {dstate, dforce, dforcerate};
}
