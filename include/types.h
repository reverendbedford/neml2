#pragma once

#include <cstddef>
#include <torch/torch.h>

typedef int64_t TorchSize;
typedef std::vector<TorchSize> TorchShape;
typedef torch::IntArrayRef TorchShapeRef;
typedef torch::ArrayRef<at::indexing::TensorIndex> TorchSlice;

