#pragma once

#include "models/Model.h"
#include "tensors/Scalar.h"
#include "tensors/SymR2.h"
#include "tensors/SymSymR4.h"

/// Templated base class of all external driving forces
template <typename T, bool stateful>
class Force : public Model
{
public:
  Force(const std::string & name);
};
