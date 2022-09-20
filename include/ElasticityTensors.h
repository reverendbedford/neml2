#pragma once

#include "SymSymR4.h"
#include "Scalar.h"

// We ought to be able template all these and just slice on the last
// indices to fill batched and unbatched versions

/// Fill an unbatched isotropic elasticity tensor
SymSymR4 fill_isotropic(const Scalar & E, const Scalar & nu);
