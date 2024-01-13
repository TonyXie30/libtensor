#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>
#include <type_traits>
#include <cassert>
#include <stdexcept>
#include <random>

#include <list>

namespace ts {
    // Tensor模板类定义
    template<typename T>
    class Tensor {
        static_assert(std::is_same<T, int>::value || std::is_same<T, double>::value,
                      "Tensor can only be of type int or double."); // 限制类型为int或double

    public:

        //默认构造函数
        Tensor() : Tensor(std::vector<T>(1, 0), {1}) {}

        // 构造函数-1.1
        Tensor(const std::vector<T> &data, const std::vector<size_t> &shape);

        Tensor(const std::vector<T> &data, const std::vector<size_t> &shape, size_t start_index, size_t end_index);

        // 获取Tensor形状
        std::vector<size_t> get_shape() const;

        // 访问元素（非const版本）tensor({1,2,3...})
        T &operator()(const std::vector<size_t> &indexes);

        // 访问元素（const版本）tensor({1,2,3...})
        const T &operator()(const std::vector<size_t> &indexes) const;

        // 输出张量结构及元素
        void print() const;

        // 随机初始化tensor-1.2
        static Tensor<T> rand(const std::vector<size_t> &shape);

        // 创建所有元素为0的Tensor-1.3
        static Tensor<T> zeros(const std::vector<size_t> &shape);

        // 创建所有元素为1的Tensor-1.3
        static Tensor<T> ones(const std::vector<size_t> &shape);

        // 创建所有元素为指定值的Tensor-1.3
        static Tensor<T> full(const std::vector<size_t> &shape, T value);

        // 创建二维单位矩阵-1.3
        static Tensor<T> eye(size_t size);

        // 访问切片
        Tensor<T> operator()(size_t index);

        // 访问切片2
        Tensor<T> operator()(size_t index, std::vector<size_t> slice);

        //add
        Tensor<T> operator+(Tensor<T> adder);

        Tensor <T> add(Tensor <T> adder);

        Tensor <T> add(double value);

        //sub
        Tensor<T> operator-(Tensor<T> subtractor);

        Tensor <T> sub(Tensor <T> subtractor);

        Tensor <T> sub(double value);

        Tensor <T> mul(const Tensor <T> &multiplier);

        Tensor<T> operator*(const Tensor<T> &multi);

        Tensor <T> mul(double scala);

        Tensor<T> div(const Tensor<T> &t1, const Tensor<T> &t2);

        Tensor<T> div(double scalar);

        Tensor<T> operator/(const Tensor<T> &dividend);

        Tensor<T> div(const Tensor<T> &tensor);
    private:
        std::vector<T> data_; // 存储Tensor元素

        std::vector<size_t> shape_; // 存储Tensor形状

        std::vector<size_t> strides_; // 存储各维度的步长

        // 计算多维索引对应的一维索引
        size_t calculate_index(const std::vector<size_t> &indexes) const;

        Tensor<T> sum(int dim) const;

        Tensor<double> mean(int dim);

        Tensor<T> max(int dim);

        Tensor<T> min(int dim);
    };

// 构造函数实现
    template<typename T>
    Tensor<T>::Tensor(const std::vector<T> &data, const std::vector<size_t> &shape)
            : data_(data), shape_(shape) {
        if (shape.empty()) {
            throw std::invalid_argument("Shape cannot be empty.(when create tensor)");
        }
        for (size_t i = 0; i < shape_.size(); i++) {
            if (shape[i] == 0 || shape[i] > 1000) {
                throw std::invalid_argument(
                        "Shape dimensions must be greater than zero and should not be too large.(when create tensor)");
            }
        }
        strides_.resize(shape_.size());
        size_t stride = 1;
        for (size_t i = shape_.size(); i-- > 0;) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
        if (data_.size() != stride) {
            throw std::logic_error("Data size does not match the tensor shape.(when create tensor)");
        }
    }

    // 构造函数-1.1
    template<typename T>
    Tensor<T>::Tensor(const std::vector<T> &data, const std::vector<size_t> &shape, size_t start_index,
                      size_t end_index)
            : data_(data), shape_(shape) {
        if (shape.empty()) {
            throw std::invalid_argument("Shape cannot be empty.(when create tensor)");
        }

        // Validate the start index
        if (start_index >= data_.size()) {
            throw std::out_of_range("Start index out of range.(when create tensor)");
        }

        // Calculate strides starting from the specified index
        strides_.resize(shape_.size());
        size_t stride = 1;
        for (size_t i = shape_.size(); i-- > 0;) {
            strides_[i] = stride;
            stride *= shape_[i];
        }

        // Adjust data and shape based on the starting index
        data_ = std::vector<T>(data_.begin() + start_index, data_.begin() + end_index);
        shape_[0] = data_.size();
    }

// 获取Tensor形状实现
    template<typename T>
    std::vector<size_t> Tensor<T>::get_shape() const {
        return shape_;
    }

// 访问元素实现（非const版本）
    template<typename T>
    T &Tensor<T>::operator()(const std::vector<size_t> &indexes) {
        return data_[calculate_index(indexes)];
    }

// 访问元素实现（const版本）
    template<typename T>
    const T &Tensor<T>::operator()(const std::vector<size_t> &indexes) const {
        return data_[calculate_index(indexes)];
    }

// 计算索引实现
    template<typename T>
    size_t Tensor<T>::calculate_index(const std::vector<size_t> &indexes) const {
        if (indexes.size() != shape_.size()) {
            throw std::invalid_argument("Dimension mismatch.(when use index)");
        }
        for (size_t i = 0; i < indexes.size(); i++) {
            if (indexes[i] >= shape_[i]) {
                throw std::out_of_range("Tensor index out of range.(when use index)");
            }
        }
        size_t index = 0;
        for (size_t i = 0; i < indexes.size(); ++i) {
            index += strides_[i] * indexes[i];
        }
        return index;
    }

//输出张量结构与元素的实现
    template<typename T>
    void Tensor<T>::print() const {
        int numbers = 0;
        auto stride = strides_[strides_.size() - 2];
        for (size_t i = 0; i < data_.size(); i++) {
            numbers++;
            std::cout << data_[i] << " ";
            if (numbers == stride) {
                numbers = 0;
                std::cout << std::endl;
            }
        }

    }


//rand<>,随机初始化tensor
//double类型实现
    template<>
    Tensor<double> Tensor<double>::rand(const std::vector<size_t> &shape) {
        if (shape.empty()) {
            throw std::invalid_argument("Shape cannot be empty.(when using rand to create tensor)");
        }
        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] == 0 || shape[i] > 1000) {
                throw std::invalid_argument(
                        "Shape dimensions must be greater than zero and should not be too large.(when using rand to create tensor)");
            }
        }
        size_t total_number = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            total_number *= shape[i];
        }
        std::vector<double> data(total_number);
        std::random_device rd; // 随机数生成器种子
        std::mt19937 gen(rd()); // 基于mt19937算法的随机数生成器
        std::uniform_real_distribution<double> dis(0.0, 1.0); // 定义一个从0.0到1.0的均匀分布

        for (double &val: data) {
            val = dis(gen); // 生成一个随机数并赋值
        }
        return Tensor<double>(data, shape);
    }

//rand<>,随机初始化tensor
// int类型的实现
    template<>
    Tensor<int> Tensor<int>::rand(const std::vector<size_t> &shape) {
        if (shape.empty()) {
            throw std::invalid_argument("Shape cannot be empty.(when using rand to create tensor)");
        }
        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] == 0 || shape[i] > 1000) {
                throw std::invalid_argument(
                        "Shape dimensions must be greater than zero and should not be too large.(when using rand to create tensor)");
            }
        }
        size_t total_number = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            total_number *= shape[i];
        }
        std::vector<int> data(total_number);
        std::random_device rd; // 随机数生成器种子
        std::mt19937 gen(rd()); // 基于mt19937算法的随机数生成器
        std::uniform_int_distribution<int> dis(0, 100); // 定义一个从0到100的均匀分布

        for (int &val: data) {
            val = dis(gen); // 生成一个随机数并赋值
        }
        return Tensor<int>(data, shape);
    }

// 创建所有元素为0的Tensor
    template<typename T>
    Tensor<T> Tensor<T>::zeros(const std::vector<size_t> &shape) {
        if (shape.empty()) {
            throw std::invalid_argument("Shape cannot be empty.(when using zeros to create tensor)");
        }
        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] == 0 || shape[i] > 1000) {
                throw std::invalid_argument(
                        "Shape dimensions must be greater than zero and should not be too large.(when using zeros to create tensor)");
            }
        }
        size_t total_number = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            total_number *= shape[i];
        }
        return Tensor<T>(std::vector<T>(total_number, 0), shape);
    }

// 创建所有元素为1的Tensor
    template<typename T>
    Tensor<T> Tensor<T>::ones(const std::vector<size_t> &shape) {
        if (shape.empty()) {
            throw std::invalid_argument("Shape cannot be empty.(when using ones to create tensor)");
        }
        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] == 0 || shape[i] > 1000) {
                throw std::invalid_argument(
                        "Shape dimensions must be greater than zero and should not be too large.(when using ones to create tensor)");
            }
        }
        size_t total_number = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            total_number *= shape[i];
        }
        return Tensor<T>(std::vector<T>(total_number, 1), shape);
    }

// 创建所有元素为指定值的Tensor
    template<typename T>
    Tensor<T> Tensor<T>::full(const std::vector<size_t> &shape, T value) {
        if (shape.empty()) {
            throw std::invalid_argument("Shape cannot be empty.(when using full to create tensor)");
        }

        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] == 0 || shape[i] > 1000) {
                throw std::invalid_argument(
                        "Shape dimensions must be greater than zero and should not be too large.(when using full to create tensor)");
            }
        }
        size_t total_number = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            total_number *= shape[i];
        }
        return Tensor<T>(std::vector<T>(total_number, value), shape);
    }

// 创建单位矩阵，发现需要限制这些张量的形状大小（不清楚怎么处理），尝试输入负数会导致尺寸爆炸。
    template<typename T>
    Tensor<T> Tensor<T>::eye(size_t size) {
        if (size == 0 || size > 1000) {
            throw std::invalid_argument(
                    "Shape dimensions must be greater than zero and should not be too large.(when using eye to create tensor)");
        }
        Tensor<T> mytensor = Tensor<T>::zeros({size, size});
        for (size_t i = 0; i < size; i++) {
            mytensor({i, i}) = 1;
        }
        return mytensor;
    }

// Tensor operations

    // 访问切片
    template<typename T>
    Tensor<T> Tensor<T>::operator()(size_t index) {
        size_t shape = shape_[0];

        if (index >= shape) {
            throw std::out_of_range("Invalid index for slicing.");
        }

        // Calculate the starting index for the sliced dimension
        size_t start_index = index * strides_[0];
        size_t end_index = start_index + strides_[0];
        // Create a new tensor with the same data but a different shape
        Tensor<T> sliced_tensor(data_, shape_, start_index, end_index);

        return sliced_tensor;
    }

    //切片方式2
    template<typename T>
    Tensor<T> Tensor<T>::operator()(size_t index, std::vector<size_t> slice) {
        size_t shape = shape_[0];

        if (index >= shape) {
            throw std::out_of_range("Invalid index for slicing.");
        }

        if (slice.size()!=2){
            throw std::invalid_argument("not a valid slice input");
        }

        if (slice[0]<0&&slice[1]>strides_[0]){
            throw std::invalid_argument("Index out of bound");
        }
        size_t start = slice[0];
        size_t end = slice[1];

        // Calculate the starting index for the sliced dimension
        size_t start_index = index * strides_[0];
        // Now calculate the detailed slice
        size_t end_index = start_index + end;
        start_index += start;
        // Create a new tensor with the same data but a different shape
        Tensor<T> sliced_tensor(data_, shape_, start_index, end_index);

        return sliced_tensor;
    }






}


#endif // TENSOR_H
