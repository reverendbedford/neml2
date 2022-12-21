#include "models/solid_mechanics/StressMeasure.h"

namespace neml2
{
StressMeasure::StressMeasure(const std::string & name)
  : SecDerivModel(name),
    overstress(declareInputVariable<SymR2>({"state", "overstress"})),
    stress_measure(declareOutputVariable<Scalar>({"state", "stress_measure"}))
{
  setup();
}
} // namespace neml2
