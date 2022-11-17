#include "models/RateForm.h"

StateInfo
RateForm::output() const
{
  return state().add_suffix("_rate");
}
