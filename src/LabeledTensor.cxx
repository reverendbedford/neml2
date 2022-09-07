#include "LabeledTensor.h"

LabeledTensor::LabeledTensor() :
    torch::Tensor()
{

}

LabeledTensor::LabeledTensor(const torch::Tensor & tensor) :
    torch::Tensor(tensor)
{

}

LabeledTensor::LabeledTensor(const torch::Tensor & tensor, 
                             std::map<std::string,TorchSlice> labels) :
    torch::Tensor(tensor), labels_(labels)
{

}

void LabeledTensor::add_label(std::string label, TorchSlice indices)
{
  labels_.insert({label, indices});
}


torch::Tensor LabeledTensor::get(std::string label)
{
  return torch::Tensor::index(labels_.at(label));
}
