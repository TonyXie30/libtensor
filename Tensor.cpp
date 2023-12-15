#include "Tensor.h"
#include <cassert>

Tensor::Tensor(const std::vector<double>& data, const std::vector<size_t>& shape)
    : data_(data), shape_(shape) {
    strides_.resize(shape_.size());
    size_t stride = 1;
    for (size_t i = shape_.size(); i-- > 0;) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
    assert(data_.size() == stride);
}

std::vector<size_t> Tensor::get_shape() const {
    return shape_;
}

double& Tensor::operator()(const std::vector<size_t>& indexes) {
    return data_[calculate_index(indexes)];
}

const double& Tensor::operator()(const std::vector<size_t>& indexes) const {
    return data_[calculate_index(indexes)];
}

size_t Tensor::calculate_index(const std::vector<size_t>& indexes) const {
    size_t index = 0;
    for (size_t i = 0; i < indexes.size(); ++i) {
        assert(indexes[i] < shape_[i]); // Ensure indexes are valid
        index += strides_[i] * indexes[i];
    }
    return index;
}
