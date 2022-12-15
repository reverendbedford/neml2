#pragma once

#include "models/ImplicitModel.h"

namespace neml2
{
/// Defines the residual of first order time integration as
/// r = s_np1 - s_n - s_dot * (t_np1 - t_n)
class ImplicitTimeIntegration : public ImplicitModel
{
public:
  ImplicitTimeIntegration(const std::string & name, std::shared_ptr<Model> rate);

  // Define the nonlinear system we are solving for
  virtual void set_residual(BatchTensor<1> x, BatchTensor<1> r, BatchTensor<1> * J = nullptr) const;

  const LabeledAxisAccessor time;
  const LabeledAxisAccessor time_n;
  const LabeledAxisAccessor resid;

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  const Model & _rate;
};
} // namespace neml2
