#include <vector>
#include <stdexcept>
#include "Tensor.h"

using namespace ts;

//Math operations

// add
template<typename T>
Tensor<T> Tensor<T>::add(Tensor<T> adder) {
    if (shape_ != adder.get_shape()) {
        throw std::invalid_argument("Cannot add tensors with different shapes.");
    }

    std::vector<T> result_data;
    result_data.reserve(data_.size());

    for (size_t i = 0; i < data_.size(); ++i) {
        result_data.push_back(data_[i] + adder(i));
    }

    return Tensor<T>(result_data, shape_);
}

// add (global function)
template<typename T>
Tensor<T> add(Tensor<T> a1, Tensor<T> a2) {
    if (a1.get_shape() != a2.get_shape()) {
        throw std::invalid_argument("Cannot add tensors with different shapes.");
    }

    std::vector<T> result_data;
    result_data.reserve(a1.size());

    for (size_t i = 0; i < a1.size(); ++i) {
        result_data.push_back(a1(i) + a2(i));
    }

    return Tensor<T>(result_data, a1.get_shape());
}

// operator+
template<typename T>
Tensor<T> Tensor<T>::operator+(Tensor<T> adder) {
    return add(adder);
}

// add with mixed types
template<typename T>
Tensor<T> add(Tensor<T> t, double value) {
    std::vector<T> result_data;
    result_data.reserve(t.size());

    for (size_t i = 0; i < t.size(); ++i) {
        result_data.push_back(t(i) + static_cast<T>(value));
    }

    return Tensor<T>(result_data, t.get_shape());
}

// add with double
template<typename T>
Tensor<T> Tensor<T>::add(double value) {
    std::vector<T> result_data;
    result_data.reserve(data_.size());

    for (size_t i = 0; i < data_.size(); ++i) {
        result_data.push_back(data_[i] + static_cast<T>(value));
    }

    return Tensor<T>(result_data, shape_);
}

// sub
template<typename T>
Tensor<T> Tensor<T>::sub(Tensor<T> subtractor) {
    if (shape_ != subtractor.get_shape()) {
        throw std::invalid_argument("Cannot subtract tensors with different shapes.");
    }

    std::vector<T> result_data;
    result_data.reserve(data_.size());

    for (size_t i = 0; i < data_.size(); ++i) {
        result_data.push_back(data_[i] - subtractor(i));
    }

    return Tensor<T>(result_data, shape_);
}

// sub (global function)
template<typename T>
Tensor<T> sub(Tensor<T> a1, Tensor<T> a2) {
    if (a1.get_shape() != a2.get_shape()) {
        throw std::invalid_argument("Cannot subtract tensors with different shapes.");
    }

    std::vector<T> result_data;
    result_data.reserve(a1.size());

    for (size_t i = 0; i < a1.size(); ++i) {
        result_data.push_back(a1(i) - a2(i));
    }

    return Tensor<T>(result_data, a1.get_shape());
}

// operator-
template<typename T>
Tensor<T> Tensor<T>::operator-(Tensor<T> subtractor) {
    return sub(subtractor);
}

// sub with mixed types
template<typename T>
Tensor<T> sub(Tensor<T> t, double value) {
    std::vector<T> result_data;
    result_data.reserve(t.size());

    for (size_t i = 0; i < t.size(); ++i) {
        result_data.push_back(t(i) - static_cast<T>(value));
    }

    return Tensor<T>(result_data, t.get_shape());
}

// sub with double
template<typename T>
Tensor<T> Tensor<T>::sub(double value) {
    std::vector<T> result_data;
    result_data.reserve(data_.size());

    for (size_t i = 0; i < data_.size(); ++i) {
        result_data.push_back(data_[i] - static_cast<T>(value));
    }

    return Tensor<T>(result_data, shape_);
}

// Mul for high-dimensional tensors
template<typename T>
Tensor<T> Tensor<T>::mul(const Tensor<T> &multiplier){
    // Check if dimensions are compatible for multiplication
    for (size_t i = 0; i < shape_.size() - 1; ++i) {
        if (shape_[i] != multiplier.shape_[i]) {
            throw std::invalid_argument("Incompatible dimensions for multiplication");
        }
    }

    // Calculate the resulting shape
    std::vector<size_t> result_shape;
    for (size_t i = 0; i < shape_.size() - 1; ++i) {
        result_shape.push_back(shape_[i]);
    }
    result_shape.push_back(multiplier.shape_.back());

    // Create the result Tensor
    Tensor<T> result = zeros(result_shape);

    // Perform tensor multiplication
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < multiplier.shape_.back(); ++j) {
            // Initialize the result element
            T element = 0;

            // Iterate over the common dimension
            for (size_t k = 0; k < shape_.back(); ++k) {
                element += data_[calculate_index({i, k})] * multiplier({k, j});
            }

            // Set the result element
            result({i, j}) = element;
        }
    }

    return result;
}

// High-dimensional tensor multiplication
template<typename T>
Tensor<T> mul(const Tensor<T> &t1, const Tensor<T> &t2) {
    return t1.mul(t2);
}

// Override * operator for high-dimensional tensor multiplication
template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> &multi){
    return mul(*this, multi);
}

// High-dimensional tensor multiplication with a scalar
template<typename T>
Tensor<T> mul(const Tensor<T> &tensor, double scala) {
    // Create the result Tensor
    Tensor<T> result = tensor;

    // Perform scalar multiplication
    for (size_t i = 0; i < tensor.data_.size(); ++i) {
        result.data_[i] *= scala;
    }

    return result;
}

// Scalar multiplication
template<typename T>
Tensor<T> Tensor<T>::mul(double scala){
    return mul(*this, scala);
}


//divide operation
