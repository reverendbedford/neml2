#pragma once

#include "tensors/LabeledTensor.h"

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

  /// Add another vector into this vector.
  /// The item set of the other vector must be a subset of this vector's item set.
  void assemble(const LabeledVector & other);

  /// Promote to LabeledMatrix with arbitrary outer product
  LabeledMatrix outer(const LabeledVector & other) const;

  /// Write to a stream
  void write(std::ostream & os, std::string delimiter, TorchSize batch, bool header = false) const;
};
