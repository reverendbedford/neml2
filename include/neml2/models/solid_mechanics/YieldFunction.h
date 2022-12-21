#pragma once

#include "neml2/models/SecDerivModel.h"
#include "neml2/models/solid_mechanics/StressMeasure.h"
#include "neml2/tensors/LabeledAxis.h"

namespace neml2
{
/// Parent class for all yield functions
class YieldFunction : public SecDerivModel
{
public:
  /// Calculate yield function knowing the corresponding hardening model
  YieldFunction(const std::string & name,
                const std::shared_ptr<StressMeasure> & sm,
                Scalar s0,
                bool with_isotropic_hardening,
                bool with_kinematic_hardening);

  const LabeledAxisAccessor mandel_stress;
  const LabeledAxisAccessor yield_function;
  const LabeledAxisAccessor isotropic_hardening;
  const LabeledAxisAccessor kinematic_hardening;
  /// Stress measure
  const StressMeasure & stress_measure;

protected:
  /// The value of the yield function
  virtual void
  set_value(LabeledVector in, LabeledVector out, LabeledMatrix * dout_din = nullptr) const;

  /// The derivative of the yield function w.r.t. hardening variables
  virtual void set_dvalue(LabeledVector in,
                          LabeledMatrix dout_din,
                          LabeledTensor<1, 3> * d2out_din2 = nullptr) const;

private:
  LabeledVector make_stress_measure_input(LabeledVector in) const;

protected:
  /// Yield stress
  Scalar _s0;

  /// @{
  /// Whether we include isotropic and/or kinematic hardening
  const bool _with_isotropic_hardening;
  const bool _with_kinematic_hardening;
  /// @}
};
} // namespace neml2
