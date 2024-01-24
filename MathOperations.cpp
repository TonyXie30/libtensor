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
        std::vector<T> result;

        // Perform scalar division
        for (size_t i = 0; i < tensor.data_->size(); ++i) {
            result.push_back((double)(*this->getData())[i]*scala);
        }

        return Tensor<T>(result,shape_);
    }

// Divide operation for element-wise division
    template<typename T>
    Tensor<double> Tensor<T>::div(const Tensor<T> &tensor) {
        // Check if dimensions are compatible for element-wise division
        if (shape_ != tensor.get_shape()) {
            throw std::invalid_argument("Incompatible dimensions for element-wise division");
        }

        // Create the result Tensor
        std::vector<double> result;

        // Perform element-wise division
        for (size_t i = 0; i < data_->size(); ++i) {
            if ((*tensor.data_)[i] == 0) {
                throw std::invalid_argument("Division by zero");
            }
            result.push_back((double)(*data_)[i] / (*tensor.data_)[i]);
        }

        return Tensor<double>(result,shape_);
    }

// Override / operator for high-dimensional tensor division
    template<typename T>
    Tensor<double> Tensor<T>::operator/(const Tensor<T> &dividend) {
        return this->div(dividend);
    }

// Scalar division
    template<typename T>
    Tensor<double> Tensor<T>::div(double scalar) {
        Tensor<T> &tensor = *this;
        if (scalar == 0) {
            throw std::invalid_argument("Division by zero");
        }

        // Create the result Tensor
        std::vector<double> result;

        // Perform scalar division
        for (size_t i = 0; i < tensor.data_->size(); ++i) {
            result.push_back((double)(*this->getData())[i]/scalar);
        }

        return Tensor<double>(result,shape_);
    }

// 1. pointwise operations end

// 2. Reduction operations

// Sum along a specified dimension for member function
    template<typename T>
    Tensor<double> Tensor<T>::sum(int dim) const {
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
        for (int dimension:this->get_shape()) {
            size *= dimension;
        }
        std::vector<double> data;
        recursiveSum((*this->getData()),this->get_shape(),0,size-1,dim,0,data);

        return Tensor<double> (data,result_shape);;

    }


// Mean along a specified dimension for member function
    template<typename T>
    Tensor<double> Tensor<T>::mean(int dim) const {
        if (dim < 0 || static_cast<size_t>(dim) >= shape_.size()) {
            throw std::out_of_range("Invalid dimension for mean operation.");
        }

        // 初始化结果数组
        std::vector<size_t> result_shape;
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i != static_cast<size_t>(dim)) {
                result_shape.push_back(shape_[i]);
            }
        }

        size_t size = 1;
        for (int dimension:shape_) {
            size *= dimension;
        }
        std::vector<double> data;
        recursiveMean((*this->getData()),this->get_shape(),0,size-1,dim,0,data);

        return Tensor<double> (data,result_shape);;
    }

    template<typename T>
    Tensor<double> Tensor<T>::max(int dim) const {
        if (dim < 0 || static_cast<size_t>(dim) >= this->get_shape().size()) {
            throw std::out_of_range("Invalid dimension for max operation.");
        }

        // 初始化结果数组
        std::vector<size_t> result_shape;
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i != static_cast<size_t>(dim)) {
                result_shape.push_back(shape_[i]);
            }
        }

        size_t size = 1;
        for (int dimension:shape_) {
            size *= dimension;
        }
        std::vector<double> data;
        recursiveMax((*this->getData()),this->get_shape(),0,size-1,dim,0,data);

        return Tensor<double> (data,result_shape);
    }

    template<typename T>
    Tensor<double> Tensor<T>::min(int dim) const {
        if (dim < 0 || static_cast<size_t>(dim) >= this->get_shape().size()) {
            throw std::out_of_range("Invalid dimension for max operation.");
        }

        // 初始化结果数组
        std::vector<size_t> result_shape;
        for (size_t i = 0; i < shape_.size(); ++i) {
            if (i != static_cast<size_t>(dim)) {
                result_shape.push_back(shape_[i]);
            }
        }

        size_t size = 1;
        for (int dimension:shape_) {
            size *= dimension;
        }
        std::vector<double> data;
        recursiveMin((*this->getData()),this->get_shape(),0,size-1,dim,0,data);

        return Tensor<double> (data,result_shape);;
    }

// 2. Reduction operations end


// 3. Comparison operations

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