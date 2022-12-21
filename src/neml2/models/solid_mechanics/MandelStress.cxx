#include "neml2/models/solid_mechanics/MandelStress.h"

namespace neml2
{
MandelStress::MandelStress(const std::string & name)
  : Model(name),
    cauchy_stress(declareInputVariable<SymR2>({"state", "cauchy_stress"})),
    mandel_stress(declareOutputVariable<SymR2>({"state", "mandel_stress"}))
{
  setup();
}
} // namespace neml2
