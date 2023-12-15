#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>

class Tensor {
public:
    Tensor(const std::vector<double>& data, const std::vector<size_t>& shape);
    std::vector<size_t> get_shape() const;
    double& operator()(const std::vector<size_t>& indexes);
    const double& operator()(const std::vector<size_t>& indexes) const;

private:
    std::vector<double> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t calculate_index(const std::vector<size_t>& indexes) const;
};

#endif // TENSOR_H
