#pragma once

#include "tensors/LabeledTensor.h"

namespace neml2
{
class LabeledVector;

/// A labeled matrix
class LabeledMatrix : public LabeledTensor<1, 2>
{
public:
  using LabeledTensor<1, 2>::LabeledTensor;

  /// Conversion from LabeledTensor
  LabeledMatrix(const LabeledTensor<1, 2> & other);

  /// Construct from two already-setup `LabeledVector`s (can infer batch)
  LabeledMatrix(const LabeledVector & A, const LabeledVector & B);

  /// Since we assume a flat batch for now, we can define this convenient method to retrive the single batch size.
  TorchSize batch_size() const { return tensor().batch_sizes()[0]; }

  /// Add another matrix into this matrix.
  /// The item set of the other matrix must be a subset of this matrix's item set.
  void accumulate(const LabeledMatrix & other, bool recursive = true);

  /// Fill another matrix into this matrix.
  /// The item set of the other matrix must be a subset of this matrix's item set.
  void fill(const LabeledMatrix & other, bool recursive = true);

  /// Chain rule product of two derivatives
  LabeledMatrix chain(const LabeledMatrix & other) const;

  /// Invert a LabeledMatrix for use in an implicit function derivative
  LabeledMatrix inverse() const;

  /// Write to a stream
  void write(std::ostream & os, std::string delimiter, TorchSize batch, bool header = false) const;
};
} // namespace neml2
