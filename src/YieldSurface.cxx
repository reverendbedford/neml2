#include "YieldSurface.h"
#include "StateInfo.h"

StateInfo
YieldSurface::interface() const
{
  StateInfo interface;
  StateInfo stress;
  stress.add<SymR2>("stress");
  interface.add_substate("stress_interface", stress);
  interface.add_substate("hardening_interface", hardening_interface());
  return interface;
}
