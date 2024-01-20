#include <vector>
#include <stdexcept>
#include "cmath"
#include "algorithm"
#include "Tensor.h"

namespace ts
{
    // 1. Pointwise operations

    // add
    template <typename T>
    Tensor<T> Tensor<T>::add(Tensor<T> adder)
    {
        if (shape_ != adder.get_shape())
        {
            throw std::invalid_argument("Cannot add tensors with different shapes.");
        }

        std::vector<T> result_data;
        result_data.reserve(data_->size());

        for (size_t i = 0; i < data_->size(); ++i) {
            result_data.push_back((*data_)[i] + (*adder.data_)[i]);
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

// Divide operation for high-dimensional tensor division
    template<typename T>
    Tensor<T> div(const Tensor<T> &t1, const Tensor<T> &t2) {
        // Check if dimensions are compatible for tensor division
        return t1.div(t2);
    }

// Override / operator for high-dimensional tensor division
    template<typename T>
    Tensor<T> Tensor<T>::operator/(const Tensor<T> &dividend) {
        return this->div(dividend);
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
        for (size_t i = 0; i < tensor.data_->size(); ++i) {
            (*result.data_)[i] /= scalar;
        }

        return result;
    }


    template<typename T>
    Tensor<T> log(const Tensor<T> &tensor) {
        // Create the result Tensor
        Tensor<T> result = tensor;

        // Perform element-wise logarithm
        for (size_t i = 0; i < tensor.data_->size(); ++i) {
            result.data_[i] = std::log(tensor.data_[i]);
        }

        return result;
    }

// 1. pointwise operations end

// 2. Reduction operations

// Sum along a specified dimension for member function
    template<typename T>
    Tensor<T> Tensor<T>::sum(int dim) const {
        if (dim < 0 || static_cast<size_t>(dim) >= shape_.size()) {
            throw std::out_of_range("Invalid dimension for sum operation.");
        }

        // Calculate the new shape after summing along the specified dimension
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

        // Calculate the new data vector after summing along the specified dimension
        std::vector<T> new_data;
        new_data.reserve(data_->size() / shape_[dim]);

        for (size_t i = 0; i < num_slices; ++i) {
            T sum = 0;

            for (size_t j = 0; j < slice_size; ++j) {
                size_t index = i * shape_[dim] * slice_size + j;
                sum += (*data_)[index];
            }

            new_data.push_back(sum);
        }

        return Tensor<T>(new_data, new_shape);
    }

// Sum along a specified dimension for non-member function
    template<typename T>
    Tensor<T> sum(const Tensor<T> &tensor, int dim) {
        return tensor.sum(dim);
    }

// Mean along a specified dimension for member function
    template<typename T>
    Tensor<double> Tensor<T>::mean(int dim) const{
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

// Mean along a specified dimension for non-member function
    template<typename T>
    Tensor<double> mean(const Tensor<T> &tensor, int dim) {
        return tensor.mean(dim);
    }

    template<typename T>
    Tensor<T> max(const Tensor<T> &tensor, int dim){
        return tensor.max(dim);
    }

    template<typename T>
    Tensor<T> Tensor<T>::max(int dim) const{
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
    Tensor<T> Tensor<T>::min(int dim) const{
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

    template<typename T>
    Tensor<T> min(const Tensor<T> &tensor, int dim) {
        return tensor.min(dim);
    }

// 2. Reduction operations end


// 3. Comparison operations
    template<typename T>
    std::vector<size_t> calculate_broadcast_shape(const Tensor<T>& t1, const Tensor<T>& t2) {
        // Determine the maximum dimensionality
        size_t max_dim = std::max(t1.get_shape().size(), t2.get_shape().size());
        std::vector<size_t> newT1 = t1.get_shape();
        std::reverse(newT1.begin(),newT1.end());
        std::vector<size_t> newT2 = t2.get_shape();
        std::reverse(newT2.begin(),newT2.end());
        // Calculate the new shape with inserted dimensions
        std::vector<size_t> new_shape;
        for (size_t i = 0; i < max_dim; ++i) {
            size_t dim_t1 = (i < t1.get_shape().size()) ? newT1[i] : 1;
            size_t dim_t2 = (i < t2.get_shape().size()) ? newT2[i] : 1;
            new_shape.push_back(std::max(dim_t1, dim_t2));
        }
        std::reverse(new_shape.begin(),new_shape.end());
        return new_shape;
    }

    bool is_broadcastable(const std::vector<size_t>& t1_shape, const std::vector<size_t>& t2_shape) {
        std::vector<size_t> newT1 = t1_shape;
        std::reverse(newT1.begin(),newT1.end());
        std::vector<size_t> newT2 = t2_shape;
        std::reverse(newT2.begin(),newT2.end());

        if (t1_shape.empty()||t2_shape.empty())
            return false;

        unsigned long long total_size = std::max(newT1.size(),newT2.size());
        for (int i = 0; i < total_size; ++i) {
            size_t t1_num = i>=newT1.size()-1?1:newT1[i];
            size_t t2_num = i>=newT2.size()-1?1:newT2[i];
            if ((t1_num!=1&&t2_num!=1)&&(t1_num!=t2_num))
                return false;
        }
        return true;
    }

    template<typename T>
    Tensor<T> broadcast(const Tensor<T>& input, const Tensor<T>& other) {
        // Check if input shape is compatible with target shape for broadcasting
        if (!is_broadcastable(input.get_shape(), other.get_shape())) {
            throw std::invalid_argument("Cannot broadcast tensors with incompatible shapes.");
        }

        // Create a new shape for the broadcasted tensor
        std::vector<size_t> broadcasted_shape = calculate_broadcast_shape(input,other);

        // Calculate the size of the broadcasted data
        size_t broadcast_size = 1;
        for (size_t dim : broadcasted_shape) {
            broadcast_size *= dim;
        }
        // Create a new data vector for the broadcasted tensor
        std::vector<T> broadcasted_data;
        broadcasted_data.reserve(broadcast_size);

        // Iterate over the original data and copy it to the broadcasted data
        std::vector<size_t> current_index(broadcasted_shape.size(), 0);
        broadcasted_data = broadcastExtend(input,broadcasted_shape);

        // Create and return the broadcasted tensor
        return Tensor<T>(broadcasted_data, broadcasted_shape);
    }

    template<typename T>
    std::vector<T> broadcastExtend(const Tensor<T>& input,std::vector<size_t> &target_shape){
        size_t input_dims = input.get_shape().size();
        size_t target_dims = target_shape.size();

        // Calculate the number of dimensions for broadcasting
        size_t broadcast_dims = std::max(input_dims, target_dims);

        // Calculate the size of the broadcasted data
        size_t broadcast_size = 1;
        for (size_t dim : target_shape) {
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

//    //eq
//    template <typename T>
//    Tensor<int> eq(Tensor<T> &t1,Tensor<T> &t2){
//        if (t1.get_shape().size() != t2.get_shape().size()) {
//            throw std::invalid_argument("Cannot compare tensors with different shapes.");
//        }
//
//        // Calculate the new shape after broadcasting
//        std::vector<size_t> new_shape;
//        for (size_t i = 0; i < t1.get_shape().size(); ++i) {
//            new_shape.push_back(std::max(t1.get_shape()[i], t2.get_shape()[i]));
//        }
//
//        // Broadcast the tensors to the new shape
//        Tensor<T> broadcasted_t1 = broadcast(t1, t2);
//        Tensor<T> broadcasted_t2 = broadcast(t2, t1);
//
//        // Element-wise equality comparison
//        std::vector<int> result_data;
//        result_data.reserve(broadcasted_t1.getData()->size());
//
//        for (size_t i = 0; i < broadcasted_t1.getData()->size(); ++i) {
//            result_data.push_back(broadcasted_t1({i}) == broadcasted_t2({i})?1:0);
//        }
//
//        return Tensor<int>(result_data, new_shape);
//    }
//
//    template <typename T>
//    Tensor<int> Tensor<T>::eq(Tensor<T>& tensor){
//        return ts::eq(*this,tensor);
//    }
//
//    template <typename T>
//    Tensor<int> Tensor<T>::operator==(Tensor<T>& tensor){
//        return ts::eq(*this,tensor);
//    }
//
//    template <typename T>
//    Tensor<int> ne(Tensor<T> &t1,Tensor<T> &t2){
//        Tensor<int> result = ts::eq(t1,t2);
//        for (int i = 0; i < result.getData()->size(); ++i) {
//            (*result.getData())[i] = !(*result.getData())[i];
//        }
//        return result;
//    }
//
//    template <typename T>
//    Tensor<int> Tensor<T>::ne(Tensor<T> &tensor){
//        return ts::ne(*this,tensor);
//    }
//
//    template <typename T>
//    Tensor<int> Tensor<T>::operator!=(Tensor<T> &tensor){
//        return ts::ne(*this,tensor);
//    }
//
//
//    template <typename T>
//    Tensor<int> lt(const Tensor<T> &t1, const Tensor<T> &t2) {
//        return elementwise_comparison(t1, t2, [](const T& a, const T& b) { return a < b; });
//    }
//
//    template <typename T>
//    Tensor<int> le(const Tensor<T> &t1, const Tensor<T> &t2) {
//        return elementwise_comparison(t1, t2, [](const T& a, const T& b) { return a <= b; });
//    }
//
//    template <typename T>
//    Tensor<int> gt(const Tensor<T> &t1, const Tensor<T> &t2) {
//        return elementwise_comparison(t1, t2, [](const T& a, const T& b) { return a > b; });
//    }
//
//    template <typename T>
//    Tensor<int> ge(const Tensor<T> &t1, const Tensor<T> &t2) {
//        return elementwise_comparison(t1, t2, [](const T& a, const T& b) { return a >= b; });
//    }
//
//    template <typename T, typename Functor>
//    Tensor<int> elementwise_comparison(const Tensor<T> &t1, const Tensor<T> &t2, Functor comparison) {
//        if (t1.get_shape().size() != t2.get_shape().size()) {
//            throw std::invalid_argument("Cannot compare tensors with different shapes.");
//        }
//
//        // Calculate the new shape after broadcasting
//        std::vector<size_t> new_shape;
//        for (size_t i = 0; i < t1.get_shape().size(); ++i) {
//            new_shape.push_back(std::max(t1.get_shape()[i], t2.get_shape()[i]));
//        }
//
//        // Broadcast the tensors to the new shape
//        Tensor<T> broadcasted_t1 = broadcast(t1, new_shape);
//        Tensor<T> broadcasted_t2 = broadcast(t2, new_shape);
//
//        // Element-wise comparison
//        std::vector<int> result_data;
//        result_data.reserve(broadcasted_t1.getData().size());
//
//        for (size_t i = 0; i < broadcasted_t1.getData().size(); ++i) {
//            result_data.push_back(comparison(broadcasted_t1.getData()[i], broadcasted_t2.getData()[i]) ? 1 : 0);
//        }
//
//        return Tensor<int>(result_data, new_shape);
//    }

// 3. Comparison operations end
}
template
class ts::Tensor<int>;

template
class ts::Tensor<double>;