#pragma once

#include <cstddef>
#include <torch/torch.h>

typedef int64_t TorchSize;
typedef std::vector<TorchSize> TorchShape;
typedef torch::IntArrayRef TorchShapeRef;
typedef std::vector<at::indexing::TensorIndex> TorchSlice;

// Find a better way to handle this...
#define TorchDefaults {torch::kFloat64}
