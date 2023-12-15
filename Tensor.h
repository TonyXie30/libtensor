#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>
#include <type_traits>
#include <cassert>
#include <stdexcept>
#include <random>
// Tensor模板类定义
template<typename T>
class Tensor {
    static_assert(std::is_same<T, int>::value || std::is_same<T, double>::value,
                  "Tensor can only be of type int or double."); // 限制类型为int或double

public:

    //默认构造函数
    Tensor() : Tensor(std::vector<T>(1, 0), {1}) {}
    
    // 构造函数-1.1
    Tensor(const std::vector<T>& data, const std::vector<size_t>& shape);

    // 获取Tensor形状
    std::vector<size_t> get_shape() const;

    // 访问元素（非const版本）tensor({1,2,3...})
    T& operator()(const std::vector<size_t>& indexes);

    // 访问元素（const版本）tensor({1,2,3...})
    const T& operator()(const std::vector<size_t>& indexes) const;

    // 随机初始化tensor-1.2
    static Tensor<T> rand(const std::vector<size_t>& shape);

    // 创建所有元素为0的Tensor-1.3
    static Tensor<T> zeros(const std::vector<size_t>& shape);

    // 创建所有元素为1的Tensor-1.3
    static Tensor<T> ones(const std::vector<size_t>& shape);

    // 创建所有元素为指定值的Tensor-1.3
    static Tensor<T> full(const std::vector<size_t>& shape, T value);
private:
    std::vector<T> data_; // 存储Tensor元素
    std::vector<size_t> shape_; // 存储Tensor形状
    std::vector<size_t> strides_; // 存储各维度的步长

    // 计算多维索引对应的一维索引
    size_t calculate_index(const std::vector<size_t>& indexes) const;
};

// 构造函数实现
template<typename T>
Tensor<T>::Tensor(const std::vector<T>& data, const std::vector<size_t>& shape)
    : data_(data), shape_(shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty.(when create tensor)");
    }
    for (size_t i = 0; i < shape_.size(); i++) {
        if (shape_[i] <= 0) { 
            throw std::invalid_argument("Shape dimensions must be greater than zero.(when create tensor)");
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

// 获取Tensor形状实现
template<typename T>
std::vector<size_t> Tensor<T>::get_shape() const {
    return shape_;
}

// 访问元素实现（非const版本）
template<typename T>
T& Tensor<T>::operator()(const std::vector<size_t>& indexes) {
    return data_[calculate_index(indexes)];
}

// 访问元素实现（const版本）
template<typename T>
const T& Tensor<T>::operator()(const std::vector<size_t>& indexes) const {
    return data_[calculate_index(indexes)];
}

// 计算索引实现
template<typename T>
size_t Tensor<T>::calculate_index(const std::vector<size_t>& indexes) const {
    if (indexes.size()!=shape_.size()){
        throw std::invalid_argument("Dimension mismatch.(when use index)");
    }
    for (size_t i = 0; i < indexes.size(); i++){
        if (indexes[i]>=shape_[i]){
            throw std::out_of_range("Tensor index out of range.(when use index)");
        }
    }
    size_t index = 0;
    for (size_t i = 0; i < indexes.size(); ++i) {
        index += strides_[i] * indexes[i];
    }
    return index;
}
//rand<>,随机舒适化tensor
//double类型实现
template<>
Tensor<double> Tensor<double>::rand(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty.(when using rand to create tensor)");
    }
    for (size_t i = 0; i < shape.size(); i++){
        if (shape[i]<=0){
            throw std::invalid_argument("Shape dimensions must be greater than zero.(when using rand to create tensor)");
        }
    }
    size_t total_number=1;
    for (size_t i = shape.size(); i-->0 ;){
        total_number*=shape[i];
    }
    std::vector<double> data(total_number);
    std::random_device rd; // 随机数生成器种子
    std::mt19937 gen(rd()); // 基于mt19937算法的随机数生成器
    std::uniform_real_distribution<double> dis(0.0, 1.0); // 定义一个从0.0到1.0的均匀分布

    for (double& val : data) {
        val = dis(gen); // 生成一个随机数并赋值
    }
    return Tensor<double>(data, shape);
}
//rand<>,随机舒适化tensor
// int类型的实现
template<>
Tensor<int> Tensor<int>::rand(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty.(when using rand to create tensor)");
    }
    for (size_t i = 0; i < shape.size(); i++){
        if (shape[i]<=0){
            throw std::invalid_argument("Shape dimensions must be greater than zero.(when using rand to create tensor)");
        }
    }
    size_t total_number=1;
    for (size_t i = shape.size(); i-->0 ;){
        total_number*=shape[i];
    }
    std::vector<int> data(total_number);
    std::random_device rd; // 随机数生成器种子
    std::mt19937 gen(rd()); // 基于mt19937算法的随机数生成器
    std::uniform_int_distribution<int> dis(0, 100); // 定义一个从0到100的均匀分布

    for (int& val : data) {
        val = dis(gen); // 生成一个随机数并赋值
    }
    return Tensor<int>(data, shape);
}
// 创建所有元素为0的Tensor
template<typename T>
Tensor<T> Tensor<T>::zeros(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty.(when using zeros to create tensor)");
    }
    for (size_t i = 0; i < shape.size(); i++){
        if (shape[i]<=0){
            throw std::invalid_argument("Shape dimensions must be greater than zero.(when using zeros to create tensor)");
        }
    }
    size_t total_number=1;
    for (size_t i = shape.size(); i-->0 ;){
        total_number*=shape[i];
    }
    return Tensor<T>(std::vector<T>(total_number, 0), shape);
}

// 创建所有元素为1的Tensor
template<typename T>
Tensor<T> Tensor<T>::ones(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty.(when using ones to create tensor)");
    }
    for (size_t i = 0; i < shape.size(); i++){
        if (shape[i]<=0){
            throw std::invalid_argument("Shape dimensions must be greater than zero.(when using ones to create tensor)");
        }
    }
    size_t total_number=1;
    for (size_t i = shape.size(); i-->0 ;){
        total_number*=shape[i];
    }
    return Tensor<T>(std::vector<T>(total_number, 1), shape);
}

// 创建所有元素为指定值的Tensor
template<typename T>
Tensor<T> Tensor<T>::full(const std::vector<size_t>& shape, T value) {
    if (shape.empty()) {
        throw std::invalid_argument("Shape cannot be empty.(when using full to create tensor)");
    }
    for (size_t i = 0; i < shape.size(); i++){
        if (shape[i]<=0){
            throw std::invalid_argument("Shape dimensions must be greater than zero.(when using full to create tensor)");
        }
    }
    size_t total_number=1;
    for (size_t i = shape.size(); i-->0 ;){
        total_number*=shape[i];
    }
    return Tensor<T>(std::vector<T>(total_number, value), shape);
}
#endif // TENSOR_H
