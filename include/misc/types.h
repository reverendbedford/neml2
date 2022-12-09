#pragma once

#include <cstddef>
#include <torch/torch.h>

namespace neml2
{
typedef int64_t TorchSize;
typedef std::vector<TorchSize> TorchShape;
typedef torch::IntArrayRef TorchShapeRef;
typedef at::indexing::TensorIndex TorchIndex;
typedef std::vector<at::indexing::TensorIndex> TorchSlice;

// Standard type for non-torch real numbers
typedef double Real;

// A small number
#define EPS 1e-15

// Find a better way to handle this...
#define TorchDefaults                                                                              \
  {                                                                                                \
    torch::TensorOptions()                                                                         \
        .dtype(torch::kFloat64)                                                                    \
        .layout(torch::kStrided)                                                                   \
        .device(torch::kCPU)                                                                       \
        .requires_grad(false)                                                                      \
  }

// The usual bool_pack trick...
template <bool...>
struct bool_pack;

// If the parameters in a pack are all true
template <bool... bs>
using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

// If the parameters in a pack are all convertible to R
template <class R, class... Ts>
using are_all_convertible = all_true<std::is_convertible<Ts, R>::value...>;
} // namespace neml2
