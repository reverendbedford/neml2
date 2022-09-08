#pragma once

#include <cstddef>
#include <torch/torch.h>

typedef size_t TorchSize;
typedef torch::IntArrayRef TorchShape;
typedef torch::ArrayRef<at::indexing::TensorIndex> TorchSlice;

