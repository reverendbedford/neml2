#include "ImplicitFunctionModel.h"
#include "RateForm.h"

#include <tuple>

class IntegratedRateFormModel : public ImplicitFunctionModel
{
public:
  IntegratedRateFormModel(RateForm & rate);
  /// Input: state_np1, forces_np1, state_n, forces_n
  virtual State value(StateInput input);
  /// Derivative of residual with respect to each input
  virtual StateDerivativeOutput dvalue(StateInput input);

  /// Setup the input form for the RateForm model
  std::tuple<StateInput, Scalar> setup_input(StateInput input);

  /// Get state information from RateForm
  virtual StateInfo state() const;

  /// Get initial_state from RateForm
  virtual void initial_state(State & state) const;

  /// Get forces information from RateForm
  virtual StateInfo forces() const;

protected:
  RateForm & _rate;
};
