#include <vector>
#include <stdexcept>
#include "cmath"
#include "Tensor.h"

namespace ts {
    // 1. Pointwise operations

// add
    template<typename T>
    Tensor<T> Tensor<T>::add(Tensor<T> adder) {
        if (shape_ != adder.get_shape()) {
            throw std::invalid_argument("Cannot add tensors with different shapes.");
        }

        std::vector<T> result_data;
        result_data.reserve(data_.size());

        for (size_t i = 0; i < data_.size(); ++i) {
            result_data.push_back(data_[i] + adder.data_[i]);
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
            result_data.push_back(data_[i] - subtractor.data_[i]);
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
    Tensor<T> Tensor<T>::mul(const Tensor<T> &multiplier) {
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
    Tensor<T> Tensor<T>::operator*(const Tensor<T> &multi) {
        return this->mul(multi);
    }

// High-dimensional tensor multiplication with a scalar
    template<typename T>
    Tensor<T> mul(Tensor<T> &tensor, double scala) {
        return tensor.mul(scala);
    }

// Scalar multiplication
    template<typename T>
    Tensor<T> Tensor<T>::mul(double scala) {
        Tensor<T> &tensor = *this;
        // Create the result Tensor
        Tensor<T> result = tensor;

        // Perform scalar multiplication
        for (size_t i = 0; i < tensor.data_.size(); ++i) {
            result.data_[i] *= scala;
        }

        return result;
    }


// Divide operation for element-wise division
    template<typename T>
    Tensor<T> Tensor<T>::div(const Tensor<T> &tensor) {
        // Check if dimensions are compatible for element-wise division
        if (shape_ != tensor.get_shape()) {
            throw std::invalid_argument("Incompatible dimensions for element-wise division");
        }

        // Create the result Tensor
        Tensor<T> result = zeros(shape_);

        // Perform element-wise division
        for (size_t i = 0; i < data_.size(); ++i) {
            if (tensor.data_[i] == 0) {
                throw std::invalid_argument("Division by zero");
            }
            result.data_[i] = data_[i] / tensor.data_[i];
        }

        return result;
    }

// Divide operation for high-dimensional tensor division
    template<typename T>
    Tensor<T> Tensor<T>::div(const Tensor<T> &t1, const Tensor<T> &t2) {
        // Check if dimensions are compatible for tensor division
        if (t1.get_shape().back() != t2.get_shape().front()) {
            throw std::invalid_argument("Incompatible dimensions for tensor division");
        }

        // Calculate the resulting shape
        std::vector<size_t> result_shape = t1.get_shape();
        result_shape.back() = t2.get_shape().back();

        // Create the result Tensor
        Tensor<T> result = zeros(result_shape);

        // Perform tensor division
        for (size_t i = 0; i < t1.get_shape().front(); ++i) {
            for (size_t j = 0; j < t2.get_shape().back(); ++j) {
                for (size_t k = 0; k < t1.get_shape().back(); ++k) {
                    if (t2({k, j}) == 0) {
                        throw std::invalid_argument("Division by zero");
                    }
                    result({i, j}) += t1({i, k}) / t2({k, j});
                }
            }
        }

        return result;
    }

// Override / operator for high-dimensional tensor division
    template<typename T>
    Tensor<T> Tensor<T>::operator/(const Tensor<T> &dividend) {
        return div(*this, dividend);
    }

// Divide operation for scalar division
    template<typename T>
    Tensor<T> div(Tensor<T> &tensor, double scalar) {
        return tensor.div(scalar);
    }

// Scalar division
    template<typename T>
    Tensor<T> Tensor<T>::div(double scalar) {
        Tensor<T> &tensor = *this;
        if (scalar == 0) {
            throw std::invalid_argument("Division by zero");
        }

        // Create the result Tensor
        Tensor<T> result = tensor;

        // Perform scalar division
        for (size_t i = 0; i < tensor.data_.size(); ++i) {
            result.data_[i] /= scalar;
        }

        return result;
    }


//    template<typename T>
//    Tensor<T> log(const Tensor<T> &tensor) {
//        // Create the result Tensor
//        Tensor<T> result = tensor;
//
//        // Perform element-wise logarithm
//        for (size_t i = 0; i < tensor.data_.size(); ++i) {
//            result.data_[i] = std::log(tensor.data_[i]);
//        }
//
//        return result;
//    }
//
//// 1. pointwise operations end
//
//// 2. Reduction operations
//
//// Sum along a specified dimension for member function
//    template<typename T>
//    Tensor<T> Tensor<T>::sum(int dim) const {
//        if (dim < 0 || static_cast<size_t>(dim) >= shape_.size()) {
//            throw std::out_of_range("Invalid dimension for sum operation.");
//        }
//
//        // Calculate the new shape after summing along the specified dimension
//        std::vector<size_t> new_shape;
//        new_shape.reserve(shape_.size() - 1);
//
//        for (size_t i = 0; i < shape_.size(); ++i) {
//            if (static_cast<int>(i) != dim) {
//                new_shape.push_back(shape_[i]);
//            }
//        }
//
//        // Calculate the size of each slice along the specified dimension
//        size_t slice_size = shape_[dim];
//
//        // Calculate the number of slices along the specified dimension
//        size_t num_slices = data_.size() / (slice_size * shape_[dim]);
//
//        // Calculate the new data vector after summing along the specified dimension
//        std::vector<T> new_data;
//        new_data.reserve(data_.size() / shape_[dim]);
//
//        for (size_t i = 0; i < num_slices; ++i) {
//            T sum = 0;
//
//            for (size_t j = 0; j < slice_size; ++j) {
//                size_t index = i * shape_[dim] * slice_size + j;
//                sum += data_[index];
//            }
//
//            new_data.push_back(sum);
//        }
//
//        return Tensor<T>(new_data, new_shape);
//    }
//
//// Sum along a specified dimension for non-member function
//    template<typename T>
//    Tensor<T> sum(const Tensor<T> &tensor, int dim) {
//        return tensor.sum(dim);
//    }
//
//// Mean along a specified dimension for member function
//    template<typename T>
//    Tensor<double> Tensor<T>::mean(int dim) {
//        if (dim < 0 || static_cast<size_t>(dim) >= shape_.size()) {
//            throw std::out_of_range("Invalid dimension for mean operation.");
//        }
//
//        // Calculate the new shape after taking mean along the specified dimension
//        std::vector<size_t> new_shape;
//        new_shape.reserve(shape_.size() - 1);
//
//        for (size_t i = 0; i < shape_.size(); ++i) {
//            if (static_cast<int>(i) != dim) {
//                new_shape.push_back(shape_[i]);
//            }
//        }
//
//        // Calculate the size of each slice along the specified dimension
//        size_t slice_size = shape_[dim];
//
//        // Calculate the number of slices along the specified dimension
//        size_t num_slices = data_.size() / (slice_size * shape_[dim]);
//
//        // Calculate the new data vector after taking mean along the specified dimension
//        std::vector<double> new_data;
//        new_data.reserve(data_.size() / shape_[dim]);
//
//        for (size_t i = 0; i < num_slices; ++i) {
//            double sum = 0;
//
//            for (size_t j = 0; j < slice_size; ++j) {
//                size_t index = i * shape_[dim] * slice_size + j;
//                sum += static_cast<double>(data_[index]);
//            }
//
//            new_data.push_back(sum / static_cast<double>(slice_size));
//        }
//
//        return Tensor<double>(new_data, new_shape);
//    }
//
//// Mean along a specified dimension for non-member function
//    template<typename T>
//    Tensor<double> mean(const Tensor<T> &tensor, int dim) {
//        return tensor.mean(dim);
//    }
//
//    template<typename T>
//    Tensor<T> max(const Tensor<T> &tensor, int dim){
//        return tensor.max(dim);
//    }
//
//    template<typename T>
//    Tensor<T> Tensor<T>::max(int dim){
//        Tensor<T> tensor = *this;
//        if (dim < 0 || static_cast<size_t>(dim) >= tensor.get_shape().size()) {
//            throw std::out_of_range("Invalid dimension for max operation.");
//        }
//
//        // Calculate the new shape after finding max along the specified dimension
//        std::vector<size_t> new_shape;
//        new_shape.reserve(tensor.get_shape().size() - 1);
//
//        for (size_t i = 0; i < tensor.get_shape().size(); ++i) {
//            if (static_cast<int>(i) != dim) {
//                new_shape.push_back(tensor.get_shape()[i]);
//            }
//        }
//
//        // Calculate the size of each slice along the specified dimension
//        size_t slice_size = tensor.get_shape()[dim];
//
//        // Calculate the number of slices along the specified dimension
//        size_t num_slices = tensor.size() / (slice_size * tensor.get_shape()[dim]);
//
//        // Calculate the new data vector after finding max along the specified dimension
//        std::vector<T> new_data;
//        new_data.reserve(tensor.size() / tensor.get_shape()[dim]);
//
//        for (size_t i = 0; i < num_slices; ++i) {
//            T max_val = std::numeric_limits<T>::min();
//
//            for (size_t j = 0; j < slice_size; ++j) {
//                size_t index = i * tensor.get_shape()[dim] * slice_size + j;
//                max_val = std::max(max_val, tensor(index));
//            }
//
//            new_data.push_back(max_val);
//        }
//
//        return Tensor<T>(new_data, new_shape);
//    }
//
//    template<typename T>
//    Tensor<T> min(const Tensor<T> &tensor, int dim){
//        return tensor.min(dim);
//    }
//
//    template<typename T>
//    Tensor<T> Tensor<T>::min(int dim){
//        Tensor<T> tensor = *this;
//        if (dim < 0 || static_cast<size_t>(dim) >= tensor.get_shape().size()) {
//            throw std::out_of_range("Invalid dimension for min operation.");
//        }
//
//        // Calculate the new shape after finding min along the specified dimension
//        std::vector<size_t> new_shape;
//        new_shape.reserve(tensor.get_shape().size() - 1);
//
//        for (size_t i = 0; i < tensor.get_shape().size(); ++i) {
//            if (static_cast<int>(i) != dim) {
//                new_shape.push_back(tensor.get_shape()[i]);
//            }
//        }
//
//        // Calculate the size of each slice along the specified dimension
//        size_t slice_size = tensor.get_shape()[dim];
//
//        // Calculate the number of slices along the specified dimension
//        size_t num_slices = tensor.size() / (slice_size * tensor.get_shape()[dim]);
//
//        // Calculate the new data vector after finding min along the specified dimension
//        std::vector<T> new_data;
//        new_data.reserve(tensor.size() / tensor.get_shape()[dim]);
//
//        for (size_t i = 0; i < num_slices; ++i) {
//            T min_val = std::numeric_limits<T>::max();
//
//            for (size_t j = 0; j < slice_size; ++j) {
//                size_t index = i * tensor.get_shape()[dim] * slice_size + j;
//                min_val = std::min(min_val, tensor(index));
//            }
//
//            new_data.push_back(min_val);
//        }
//
//        return Tensor<T>(new_data, new_shape);
//    }
// 2. Reduction operations end


// 3. Comparison operations






// 3. Comparison operations end
}
template
class ts::Tensor<int>;

template
class ts::Tensor<double>;
template ts::Tensor<int> ts::mul(Tensor<int> &tensor, double scala);
template ts::Tensor<double> ts::mul(Tensor<double> &tensor, double scala);
template ts::Tensor<int> ts::div(Tensor<int> &tensor, double scalar);
template ts::Tensor<double> ts::div(Tensor<double> &tensor, double scalar);
//template ts::Tensor<int> ts::sum(const ts::Tensor<int> &tensor, int dim);
//template ts::Tensor<double> ts::sum(const ts::Tensor<double> &tensor, int dim);
//template ts::Tensor<double> ts::mean(const ts::Tensor<int> &tensor, int dim);
//template ts::Tensor<double> ts::mean(const ts::Tensor<double> &tensor, int dim);
//template ts::Tensor<int> ts::max(const ts::Tensor<int> &tensor, int dim);
//template ts::Tensor<double> ts::max(const ts::Tensor<double> &tensor, int dim);
//template ts::Tensor<int> ts::min(const ts::Tensor<int> &tensor, int dim);
//template ts::Tensor<double> ts::min(const ts::Tensor<double> &tensor, int dim);