#pragma once

#include <cstddef>
#include <torch/torch.h>

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
