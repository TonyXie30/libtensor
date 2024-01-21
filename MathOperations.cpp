#include <vector>
#include <stdexcept>
#include "cmath"
#include "algorithm"
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
        result_data.reserve(data_->size());

        for (size_t i = 0; i < data_->size(); ++i) {
            result_data.push_back((*data_)[i] + (*adder.data_)[i]);
        }

        return Tensor<T>(result_data, shape_);
    }

// operator+
    template<typename T>
    Tensor<T> Tensor<T>::operator+(Tensor<T> adder) {
        return this->add(adder);
    }

// add with double
    template<typename T>
    Tensor<T> Tensor<T>::add(double value) {
        std::vector<T> result_data;
        result_data.reserve(data_->size());

        for (size_t i = 0; i < data_->size(); ++i) {
            result_data.push_back((*data_)[i] + static_cast<T>(value));
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
        result_data.reserve(data_->size());

        for (size_t i = 0; i < data_->size(); ++i) {
            result_data.push_back((*data_)[i] - (*subtractor.data_)[i]);
        }

        return Tensor<T>(result_data, shape_);
    }

// operator-
    template<typename T>
    Tensor<T> Tensor<T>::operator-(Tensor<T> subtractor) {
        return this->sub(subtractor);
    }

// sub with double
    template<typename T>
    Tensor<T> Tensor<T>::sub(double value) {
        std::vector<T> result_data;
        result_data.reserve(data_->size());

        for (size_t i = 0; i < data_->size(); ++i) {
            result_data.push_back((*data_)[i] - static_cast<T>(value));
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


        // Create the result Tensor
        Tensor<T> result = zeros(shape_);

        // Perform element-wise multiplication
        for (size_t i = 0; i < data_->size(); ++i) {
            (*result.data_)[i] = (*data_)[i] * (*multiplier.getData())[i];
        }

        return result;
    }

// Override * operator for high-dimensional tensor multiplication
    template<typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T> &multi) {
        return this->mul(multi);
    }

// Scalar multiplication
    template<typename T>
    Tensor<T> Tensor<T>::mul(double scala) {
        Tensor<T> &tensor = *this;
        // Create the result Tensor
        Tensor<T> result = tensor;

        // Perform scalar multiplication
        for (size_t i = 0; i < tensor.data_->size(); ++i) {
            (*result.data_)[i] *= scala;
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
        for (size_t i = 0; i < data_->size(); ++i) {
            if ((*tensor.data_)[i] == 0) {
                throw std::invalid_argument("Division by zero");
            }
            (*result.data_)[i] = (*data_)[i] / (*tensor.data_)[i];
        }

        return result;
    }

// Override / operator for high-dimensional tensor division
    template<typename T>
    Tensor<T> Tensor<T>::operator/(const Tensor<T> &dividend) {
        return this->div(dividend);
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
        for (size_t i = 0; i < tensor.data_->size(); ++i) {
            (*result.data_)[i] /= scalar;
        }

        return result;
    }

// 1. pointwise operations end

// 2. Reduction operations

// Sum along a specified dimension for member function
    template<typename T>
    Tensor<T> Tensor<T>::sum(int dim) const {
        if (dim < 0 || dim >= shape_.size()) {
            throw std::invalid_argument("Invalid dimension for sum.");
        }

        // 初始化结果数组
        std::vector<size_t> result_shape;
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i != static_cast<size_t>(dim)) {
                result_shape.push_back(shape_[i]);
            }
        }

        size_t size = 1;
        for (int dimension:result_shape) {
            size *= dimension;
        }
        std::vector<T> data(size, 0.0);


        for (int i = 0; i < this->getData()->size(); ++i) {
            std::vector<int> indices(this->get_shape().size());
            int multiplier = 1;
            for (int j = 0; j < this->get_shape().size(); ++j) {
                indices[j] = (i / multiplier) % this->get_shape()[j];
                multiplier *= this->get_shape()[j];
            }
            indices[dim] = 0;

            int index = 0;
            multiplier = 1;
            for (int j = shape_.size()-1; j >=0 ; --j) {
                index += indices[j] * multiplier;
                multiplier *= shape_[j];
            }
            data[index] += (*this->getData())[i];
        }

        return Tensor<T> (data,result_shape);;

    }


// Mean along a specified dimension for member function
    template<typename T>
    Tensor<double> Tensor<T>::mean(int dim) const {
        if (dim < 0 || static_cast<size_t>(dim) >= shape_.size()) {
            throw std::out_of_range("Invalid dimension for mean operation.");
        }

        // Calculate the new shape after taking mean along the specified dimension
        std::vector<size_t> new_shape;
        new_shape.reserve(shape_.size() - 1);

        for (size_t i = 0; i < shape_.size(); ++i) {
            if (static_cast<int>(i) != dim) {
                new_shape.push_back(shape_[i]);
            }
        }

        // Calculate the size of each slice along the specified dimension
        size_t slice_size = shape_[dim];

        // Calculate the number of slices along the specified dimension
        size_t num_slices = data_->size() / (slice_size * shape_[dim]);

        // Calculate the new data vector after taking mean along the specified dimension
        std::vector<double> new_data;
        new_data.reserve(data_->size() / shape_[dim]);

        for (size_t i = 0; i < num_slices; ++i) {
            double sum = 0;

            for (size_t j = 0; j < slice_size; ++j) {
                size_t index = i * shape_[dim] * slice_size + j;
                sum += static_cast<double>((*data_)[index]);
            }

            new_data.push_back(sum / static_cast<double>(slice_size));
        }

        return Tensor<double>(new_data, new_shape);
    }

    template<typename T>
    Tensor<T> Tensor<T>::max(int dim) const {
        Tensor<T> tensor = *this;
        if (dim < 0 || static_cast<size_t>(dim) >= tensor.get_shape().size()) {
            throw std::out_of_range("Invalid dimension for max operation.");
        }

        // Calculate the new shape after finding max along the specified dimension
        std::vector<size_t> new_shape;
        new_shape.reserve(tensor.get_shape().size() - 1);

        for (size_t i = 0; i < tensor.get_shape().size(); ++i) {
            if (static_cast<int>(i) != dim) {
                new_shape.push_back(tensor.get_shape()[i]);
            }
        }

        // Calculate the size of each slice along the specified dimension
        size_t slice_size = tensor.get_shape()[dim];

        // Calculate the number of slices along the specified dimension
        size_t num_slices = tensor.get_shape().size() / (slice_size * tensor.get_shape()[dim]);

        // Calculate the new data vector after finding max along the specified dimension
        std::vector<T> new_data;
        new_data.reserve(tensor.get_shape().size() / tensor.get_shape()[dim]);

        for (size_t i = 0; i < num_slices; ++i) {
            T max_val = std::numeric_limits<T>::min();
            for (size_t j = 0; j < slice_size; ++j) {
                size_t index = i * tensor.get_shape()[dim] * slice_size + j;
                max_val = std::max(max_val, (*tensor.data_)[index]);
            }

            new_data.push_back(max_val);
        }

        return Tensor<T>(new_data, new_shape);
    }

    template<typename T>
    Tensor<T> Tensor<T>::min(int dim) const {
        Tensor<T> tensor = *this;
        if (dim < 0 || static_cast<size_t>(dim) >= tensor.get_shape().size()) {
            throw std::out_of_range("Invalid dimension for min operation.");
        }

        // Calculate the new shape after finding min along the specified dimension
        std::vector<size_t> new_shape;
        new_shape.reserve(tensor.get_shape().size() - 1);

        for (size_t i = 0; i < tensor.get_shape().size(); ++i) {
            if (static_cast<int>(i) != dim) {
                new_shape.push_back(tensor.get_shape()[i]);
            }
        }

        // Calculate the size of each slice along the specified dimension
        size_t slice_size = tensor.get_shape()[dim];

        // Calculate the number of slices along the specified dimension
        size_t num_slices = tensor.get_shape().size() / (slice_size * tensor.get_shape()[dim]);

        // Calculate the new data vector after finding min along the specified dimension
        std::vector<T> new_data;
        new_data.reserve(tensor.get_shape().size() / tensor.get_shape()[dim]);

        for (size_t i = 0; i < num_slices; ++i) {
            T min_val = std::numeric_limits<T>::max();

            for (size_t j = 0; j < slice_size; ++j) {
                size_t index = i * tensor.get_shape()[dim] * slice_size + j;
                min_val = std::min(min_val, (*tensor.data_)[index]);
            }

            new_data.push_back(min_val);
        }

        return Tensor<T>(new_data, new_shape);
    }

// 2. Reduction operations end


// 3. Comparison operations
    template<typename T>
    std::vector<size_t> calculate_broadcast_shape(const Tensor<T> &t1, const Tensor<T> &t2) {
        // Determine the maximum dimensionality
        size_t max_dim = std::max(t1.get_shape().size(), t2.get_shape().size());
        std::vector<size_t> newT1 = t1.get_shape();
        std::reverse(newT1.begin(), newT1.end());
        std::vector<size_t> newT2 = t2.get_shape();
        std::reverse(newT2.begin(), newT2.end());
        // Calculate the new shape with inserted dimensions
        std::vector<size_t> new_shape;
        for (size_t i = 0; i < max_dim; ++i) {
            size_t dim_t1 = (i < t1.get_shape().size()) ? newT1[i] : 1;
            size_t dim_t2 = (i < t2.get_shape().size()) ? newT2[i] : 1;
            new_shape.push_back(std::max(dim_t1, dim_t2));
        }
        std::reverse(new_shape.begin(), new_shape.end());
        return new_shape;
    }

    bool is_broadcastable(const std::vector<size_t> &t1_shape, const std::vector<size_t> &t2_shape) {
        std::vector<size_t> newT1 = t1_shape;
        std::reverse(newT1.begin(), newT1.end());
        std::vector<size_t> newT2 = t2_shape;
        std::reverse(newT2.begin(), newT2.end());

        if (t1_shape.empty() || t2_shape.empty())
            return false;

        unsigned long long total_size = std::max(newT1.size(), newT2.size());
        for (int i = 0; i < total_size; ++i) {
            size_t t1_num = i >= newT1.size() - 1 ? 1 : newT1[i];
            size_t t2_num = i >= newT2.size() - 1 ? 1 : newT2[i];
            if ((t1_num != 1 && t2_num != 1) && (t1_num != t2_num))
                return false;
        }
        return true;
    }

    template<typename T>
    Tensor<T> broadcast(const Tensor<T> &input, const Tensor<T> &other) {
        // Check if input shape is compatible with target shape for broadcasting
        if (!is_broadcastable(input.get_shape(), other.get_shape())) {
            throw std::invalid_argument("Cannot broadcast tensors with incompatible shapes.");
        }

        // Create a new shape for the broadcasted tensor
        std::vector<size_t> broadcasted_shape = calculate_broadcast_shape(input, other);

        // Calculate the size of the broadcasted data
        size_t broadcast_size = 1;
        for (size_t dim: broadcasted_shape) {
            broadcast_size *= dim;
        }
        // Create a new data vector for the broadcasted tensor
        std::vector<T> broadcasted_data;
        broadcasted_data.reserve(broadcast_size);

        // Iterate over the original data and copy it to the broadcasted data
        std::vector<size_t> current_index(broadcasted_shape.size(), 0);
        broadcasted_data = broadcastExtend(input, broadcasted_shape);

        // Create and return the broadcasted tensor
        return Tensor<T>(broadcasted_data, broadcasted_shape);
    }

    template<typename T>
    std::vector<T> broadcastExtend(const Tensor<T> &input, std::vector<size_t> &target_shape) {
        size_t input_dims = input.get_shape().size();
        size_t target_dims = target_shape.size();

        // Calculate the number of dimensions for broadcasting
        size_t broadcast_dims = std::max(input_dims, target_dims);

        // Calculate the size of the broadcasted data
        size_t broadcast_size = 1;
        for (size_t dim: target_shape) {
            broadcast_size *= dim;
        }

        // Calculate input strides
        std::vector<size_t> input_strides(input_dims, 1);
        for (size_t i = input_dims - 1; i < input_dims; --i) {
            input_strides[i] = (i > 0) ? input_strides[i - 1] * input.get_shape()[i] : 1;
        }

        // Calculate target strides
        std::vector<size_t> target_strides(target_dims, 1);
        for (size_t i = target_dims - 1; i < target_dims; --i) {
            target_strides[i] = (i > 0) ? target_strides[i - 1] * target_shape[i] : 1;
        }

        // Initialize vector for the broadcasted data
        std::vector<T> broadcast_data;
        broadcast_data.reserve(broadcast_size);

        // Perform the broadcasting operation
        for (size_t i = 0; i < broadcast_size; i++) {
            std::vector<size_t> input_index(input_dims, 0);
            std::vector<size_t> target_index(target_dims, 0);

            size_t remaining = i;
            for (size_t dim = 0; dim < broadcast_dims; ++dim) {
                input_index[dim] = remaining / target_strides[dim] % input.get_shape()[dim];
                target_index[dim] = remaining / target_strides[dim] % target_shape[dim];
            }

            size_t input_offset = 0;
            size_t target_offset = 0;

            for (size_t dim = 0; dim < broadcast_dims; ++dim) {
                input_offset += input_index[dim] * input_strides[dim];
                target_offset += target_index[dim] * target_strides[dim];
            }

            broadcast_data.push_back((*input.getData())[input_offset]);
        }

        return broadcast_data;
    }

    template <typename T>
    Tensor<int> Tensor<T>::eq(Tensor<T>& tensor){
        if (this->get_shape().size() != tensor.get_shape().size()) {
            throw std::invalid_argument("Cannot compare tensors with different shapes.");
        }

        // Calculate the new shape after broadcasting
        std::vector<size_t> new_shape;
        for (size_t i = 0; i < this->get_shape().size(); ++i) {
            new_shape.push_back(std::max(this->get_shape()[i], tensor.get_shape()[i]));
        }

        // Element-wise equality comparison
        std::vector<int> result_data;
        result_data.reserve(this->getData()->size());

        for (size_t i = 0; i < this->getData()->size(); ++i) {
            result_data.push_back((*this->getData())[i] == (*tensor.getData())[i]?1:0);
        }
        Tensor<int> result(result_data, new_shape);
        result.setIsBool(true);
        return result;
    }

    template <typename T>
    Tensor<int> Tensor<T>::operator==(Tensor<T>& tensor){
        return this->eq(tensor);
    }

    template <typename T>
    Tensor<int> Tensor<T>::ne(Tensor<T> &tensor){
        Tensor<int> result = this->eq(tensor);
        for (int i = 0; i < result.getData()->size(); ++i) {
            (*result.getData())[i] = !(*result.getData())[i];
        }
        return result;
    }

    template <typename T>
    Tensor<int> Tensor<T>::operator!=(Tensor<T> &tensor){
        return this->ne(tensor);
    }

    template <typename T>
    Tensor<int> Tensor<T>::lt(Tensor<T> &tensor) {
        return elementwise_comparison(*this,tensor, [](const T& a, const T& b) { return a < b; });
    }

    template <typename T>
    Tensor<int> Tensor<T>::operator<(Tensor<T> &tensor) {
        return this->lt(tensor);
    }

    template <typename T>
    Tensor<int> Tensor<T>::le(Tensor<T> &tensor) {
        return elementwise_comparison(*this,tensor, [](const T& a, const T& b) { return a <= b; });
    }

    template <typename T>
    Tensor<int> Tensor<T>::operator<=(Tensor<T> &tensor) {
        return this->le(tensor);
    }

    template <typename T>
    Tensor<int> Tensor<T>::gt(Tensor<T> &tensor) {
        return elementwise_comparison(*this,tensor, [](const T& a, const T& b) { return a > b; });
    }

    template <typename T>
    Tensor<int> Tensor<T>::operator>(Tensor<T> &tensor) {
        return this->gt(tensor);
    }

    template <typename T>
    Tensor<int> Tensor<T>::ge(Tensor<T> &tensor) {
        return elementwise_comparison(*this,tensor, [](const T& a, const T& b) { return a >= b; });
    }

    template <typename T>
    Tensor<int> Tensor<T>::operator>=(Tensor<T> &tensor) {
        return this->ge(tensor);
    }

    template <typename T, typename Functor>
    Tensor<int> elementwise_comparison(const Tensor<T> &t1, const Tensor<T> &t2, Functor comparison) {
        if (t1.get_shape().size() != t2.get_shape().size()) {
            throw std::invalid_argument("Cannot compare tensors with different shapes.");
        }

        // Calculate the new shape after broadcasting
        std::vector<size_t> new_shape;
        for (size_t i = 0; i < t1.get_shape().size(); ++i) {
            new_shape.push_back(std::max(t1.get_shape()[i], t2.get_shape()[i]));
        }

        // Element-wise comparison
        std::vector<int> result_data;
        result_data.reserve(t1.getData()->size());

        for (size_t i = 0; i < t1.getData()->size(); ++i) {
            result_data.push_back(comparison((*t1.getData())[i], (*t2.getData())[i]) ? 1 : 0);
        }

        Tensor<int> result(result_data, new_shape);
        result.setIsBool(true);
        return result;
    }

// 3. Comparison operations end
}
template
class ts::Tensor<int>;

template
class ts::Tensor<double>;