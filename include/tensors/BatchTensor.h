#pragma once

#include "types.h"
#include "utils.h"

#include <stdexcept>
#include <array>
#include <numeric>

#include <torch/torch.h>

/// The very general batched einsum
inline torch::Tensor
einsum(const std::initializer_list<torch::Tensor> & tensors,
       const std::initializer_list<c10::string_view> & indices,
       c10::string_view out_indices = "")
{
  if (tensors.size() == 0)
    throw std::runtime_error("You must provide tensors in einsum.");
  if (indices.size() != tensors.size())
    throw std::runtime_error("Number of tensor indices must match number of tensors in einsum.");

  std::string equation = "";

  // Form the equation in Einstein notation
  for (const auto & i : indices)
    equation += "..." + std::string(i) + ",";
  equation.pop_back();
  if (!out_indices.empty())
    equation += "->..." + std::string(out_indices);

  return torch::einsum(equation, tensors);
}

/// Tensor where the first index is a batch dimension
template <TorchSize N>
class BatchTensor : public torch::Tensor
{
public:
  /// Default constructor, empty with no batch dimensions
  BatchTensor();

  /// Construct from existing tensor, no batch dimensions
  BatchTensor(const torch::Tensor & tensor);

  /// Make an empty batched tensor given batch size and base size
  BatchTensor(TorchShapeRef batch_size, TorchShapeRef base_size);

  /// Make a batched tensor filled with default base tensor
  BatchTensor(const torch::Tensor & tensor, TorchShapeRef batch_size);

  /// Return the number of base dimensions
  TorchSize base_dim() const;

  /// Return the number of batch dimensions
  constexpr TorchSize batch_dim() const;

  /// Return the base size
  TorchShape base_sizes() const;

  /// Return the batch size
  TorchShape batch_sizes() const;

  /// Return the flattened storage needed just for the base indices
  TorchSize base_storage() const;

  /// Get a batch
  torch::Tensor batch_index(TorchSlice indices) const;

  /// Set a index sliced on the batch dimensions to a value
  void batch_index_put(TorchSlice indices, const torch::Tensor & other);

  /// Return an index sliced on the batch dimensions
  torch::Tensor base_index(TorchSlice indices) const;

  /// Set a index sliced on the batch dimensions to a value
  void base_index_put(TorchSlice indices, const torch::Tensor & other);

  /// Return a new view with values broadcast along the batch dimensions.
  BatchTensor<N> expand_batch(TorchShapeRef batch_size) const;

  /// Negation
  BatchTensor<N> operator-() const;

  /// Identity
  static BatchTensor<N> identity(TorchSize n);

private:
  /// Add a slice on the batch dimensions to an index
  TorchSlice make_slice(TorchSlice base) const;
};

template <TorchSize N>
BatchTensor<N>::BatchTensor()
  : BatchTensor<N>(TorchShapeRef(std::vector<TorchSize>(N, 1)), TorchShapeRef({}))
{
}

template <TorchSize N>
BatchTensor<N>::BatchTensor(const torch::Tensor & tensor)
  : torch::Tensor(tensor)
{
  // Check to make sure we can actually do this
  if (sizes().size() < N)
    throw std::runtime_error("Tensor dimension is smaller than the requested "
                             "number of batch dimensions");
}

template <TorchSize N>
BatchTensor<N>::BatchTensor(TorchShapeRef batch_size, TorchShapeRef base_size)
  : torch::Tensor(std::move(torch::zeros(utils::add_shapes(batch_size, base_size), TorchDefaults)))
{
  // Quick check to make sure batch_size is consistent with N, this could become
  // a static assertion
  if (batch_size.size() != N)
    throw std::runtime_error("Proposed batch shape does not match "
                             "the number of templated batch dimensions");
}

template <TorchSize N>
BatchTensor<N>::BatchTensor(const torch::Tensor & tensor, TorchShapeRef batch_size)
  : BatchTensor<N>(batch_size, tensor.dim() > 0 ? tensor.sizes() : 1)
{
  TorchSlice indices(tensor.sizes().size(), torch::indexing::Slice());
  base_index_put(indices, tensor);
}

template <TorchSize N>
TorchSize
BatchTensor<N>::base_dim() const
{
  return sizes().size() - batch_dim();
}

template <TorchSize N>
constexpr TorchSize
BatchTensor<N>::batch_dim() const
{
  return N;
}

template <TorchSize N>
TorchShape
BatchTensor<N>::base_sizes() const
{
  return TorchShape(sizes().begin() + N, sizes().end());
}

template <TorchSize N>
TorchShape
BatchTensor<N>::batch_sizes() const
{
  return TorchShape(sizes().begin(), sizes().begin() + N);
}

template <TorchSize N>
TorchSize
BatchTensor<N>::base_storage() const
{
  return storage_size(base_sizes());
}

template <TorchSize N>
torch::Tensor
BatchTensor<N>::batch_index(TorchSlice indices) const
{
  indices.insert(indices.end(), torch::indexing::Ellipsis);
  return this->index(indices);
}

template <TorchSize N>
void
BatchTensor<N>::batch_index_put(TorchSlice indices, const torch::Tensor & other)
{
  indices.insert(indices.end(), torch::indexing::Ellipsis);
  this->index_put_(indices, other);
}

template <TorchSize N>
torch::Tensor
BatchTensor<N>::base_index(TorchSlice indices) const
{
  indices.insert(indices.begin(), torch::indexing::Ellipsis);
  return this->index(indices);
}

template <TorchSize N>
void
BatchTensor<N>::base_index_put(TorchSlice indices, const torch::Tensor & other)
{
  indices.insert(indices.begin(), torch::indexing::Ellipsis);
  this->index_put_(indices, other);
}

template <TorchSize N>
BatchTensor<N>
BatchTensor<N>::expand_batch(TorchShapeRef batch_size) const
{
  // Quick check to make sure batch_size is consistent with N, this could become
  // a static assertion
  if (batch_size.size() != N)
    throw std::runtime_error("Proposed batch shape does not match "
                             "the number of templated batch dimensions");

  // We don't want to touch the base dimensions, so put -1 for them.
  TorchShape net(batch_size.vec());
  net.insert(net.end(), base_dim(), -1);
  return torch::expand_copy(*this, net);
}

template <TorchSize N>
BatchTensor<N>
BatchTensor<N>::operator-() const
{
  return -torch::Tensor(*this);
}

template <TorchSize N>
BatchTensor<N>
BatchTensor<N>::identity(TorchSize n)
{
  return torch::eye(n, TorchDefaults).index(TorchSlice(N, torch::indexing::None));
}

// We would like to have exact match for the basic operators to avoid ambiguity, and also to keep
// the return type as one of our supported primitive tensor types. All we need to do is to forward
// the calls to torch :)
/// @{
template <TorchSize N>
BatchTensor<N>
operator+(const BatchTensor<N> & a, const BatchTensor<N> & b)
{
  return torch::operator+(a, b);
}

template <TorchSize N>
BatchTensor<N>
operator-(const BatchTensor<N> & a, const BatchTensor<N> & b)
{
  return torch::operator-(a, b);
}
/// @}
