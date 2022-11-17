#include "models/IntegratedRateFormModel.h"
#include "state/State.h"
#include "state/StateFunction.h"
#include "state/StateInfo.h"

IntegratedRateFormModel::IntegratedRateFormModel(RateForm & rate)
  : ImplicitFunctionModel(),
    _rate(rate)
{
}

State
IntegratedRateFormModel::value(StateInput input)
{
  // I should update this to be some more general method of integration,
  // i.e. the alpha method
  State state_np1 = input[0];
  State state_n = input[2];

  auto [rate_input, dt] = setup_input(input);
  State state_rate = _rate.value(rate_input).replace_info(state());

  return state_np1.subtract(state_rate.scalar_product(dt)).subtract(state_n);
}

StateDerivativeOutput
IntegratedRateFormModel::dvalue(StateInput input)
{
  auto [rate_input, dt] = setup_input(input);
  StateDerivativeOutput derivs = _rate.dvalue(rate_input);

  StateDerivative dstate = derivs[0];
  StateDerivative dforce = derivs[1];
  StateDerivative dforcerate = derivs[2];

  State force_rate = rate_input[2];

  // Need this for some of the force derivatives
  State state_rate = _rate.value(rate_input).replace_info(state());
  State time_entry = State(forces(), dstate.batch_size());
  time_entry.set<Scalar>("time", Scalar(1, dstate.batch_size()));

  // We need to form and return the derivative of the residual with
  // respect to:
  //    1. state_np1
  //    2. forces_np1
  //    3. state_n
  //    4. forces_n
  //
  TorchSize bs = dstate.batch_size();

  // state_np1: I - dstate * dt
  auto dsnp1 = (-dstate.scalar_product(dt)).add_identity().replace_info_left(state());

  // forces_np1: sum of:
  //  -dstaterate_dforce * dt
  //  -dstaterate_dforcerate * dforcerate_dforce_np1 * dt
  //  -state_rate in the time column
  //
  //  I deal with the "simple block of dstaterate_dforce_np1 separately

  // Assemble dforcerate_dforce_np1, which is annoying
  StateDerivative dfr_dfnp1 =
      StateDerivative::id_map(
          force_rate.info(), forces(), dstate.batch_size(), {{"strain_rate", "strain"}})
          .scalar_product(1.0 / dt) +
      -force_rate.promote_outer(time_entry).scalar_product(1.0 / dt);

  //  TODO: use better operators
  auto dfnp1 = StateDerivative(state(),
                               forces(),
                               -dforce.scalar_product(dt).tensor() -
                                   dforcerate.chain(dfr_dfnp1).scalar_product(dt).tensor() -
                                   state_rate.promote_outer(time_entry).tensor());

  // state_n: -I
  StateDerivative dsn(output(), state(), bs);
  dsn = -dsn.add_identity();

  // forces_n:
  // This has all the problems as the above
  auto dfn = StateDerivative(state(),
                             forces(),
                             dforce.scalar_product(dt).tensor() * 0 // Obviously a waste
                                 + dforcerate.chain(dfr_dfnp1).scalar_product(dt).tensor() +
                                 state_rate.promote_outer(time_entry).tensor());

  return {dsnp1, dfnp1, dsn, dfn};
}

std::tuple<StateInput, Scalar>
IntegratedRateFormModel::setup_input(StateInput input)
{
  State state_np1 = input[0];
  State force_np1 = input[1];
  State state_n = input[2];
  State force_n = input[3];

  Scalar dt = force_np1.get<Scalar>("time") - force_n.get<Scalar>("time");

  // Remove time from the force rate but calculate the rate of the others
  State dforce =
      force_np1.subtract(force_n).remove("time").scalar_product(1.0 / dt).add_suffix("_rate");

  return std::tuple<StateInput, Scalar>({state_np1, force_np1, dforce}, dt);
}

StateInfo
IntegratedRateFormModel::state() const
{
  return _rate.state();
}

void
IntegratedRateFormModel::initial_state(State & state) const
{
  _rate.initial_state(state);
}

StateInfo
IntegratedRateFormModel::forces() const
{
  return _rate.forces();
}
