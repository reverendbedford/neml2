#include "State.h"

State::State(const torch::Tensor & tensor) :
    StandardBatchedLabeledTensor(tensor)
{

}

State::State(const torch::Tensor & tensor, std::map<std::string,TorchSlice>
             labels) :
    StandardBatchedLabeledTensor(tensor, labels)
{

}
