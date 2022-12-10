#pragma once

#include "models/ImplicitModel.h"
#include "solvers/NonlinearSolver.h"

namespace neml2
{
/// Update an implicit model by solving the underlying nonlinear system
class ImplicitUpdate : public Model
{
public:
  ImplicitUpdate(const std::string & name,
                 std::shared_ptr<ImplicitModel> model,
                 std::shared_ptr<NonlinearSolver> solver);

protected:
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  /// The implicit model to be updated
  ImplicitModel & _model;

  /// The nonlinear solver used to solve the nonlinear system
  const NonlinearSolver & _solver;
};
} // namespace neml2
