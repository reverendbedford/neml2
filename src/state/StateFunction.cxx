#include "state/StateFunction.h"

State
SingleStateFunction::value(StateInput input)
{
  return value(input[0]);
}

StateDerivativeOutput
SingleStateFunction::dvalue(StateInput input)
{
  return {dvalue(input[0])};
}
