#pragma once

#include "tensors/LabeledTensor.h"

namespace neml2
{
// Forward declaration
class LabeledMatrix;

/// A labeled vector
class LabeledVector : public LabeledTensor<1, 1>
{
public:
  using LabeledTensor<1, 1>::LabeledTensor;

  /// Conversion from a LabeledTensor
  LabeledVector(const LabeledTensor<1, 1> & other);

  /// Since we assume a flat batch for now, we can define this convenient method to retrive the single batch size.
  TorchSize batch_size() const { return tensor().batch_sizes()[0]; }

  /// Slice the logically 1D tensor by a single sub-axis
  LabeledVector slice(const std::string & name) const;

  /// Add another vector into this vector.
  /// The item set of the other vector must be a subset of this vector's item set.
  void accumulate(const LabeledVector & other, bool recursive = true);

  /// Fill (override) another vector into this vector.
  /// The item set of the other vector must be a subset of this vector's item set.
  void fill(const LabeledVector & other, bool recursive = true);

  /// Promote to LabeledMatrix with arbitrary outer product
  LabeledMatrix outer(const LabeledVector & other) const;

  /// Write to a stream
  void write(std::ostream & os, std::string delimiter, TorchSize batch, bool header = false) const;
};
} // namespace neml2
