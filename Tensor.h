#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>
#include <type_traits>
#include <cassert>
#include <stdexcept>
#include <random>
#include <iostream>
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

        //  cat join
        template <typename U>
        friend Tensor<U> cat(const std::vector<Tensor<U>> &tensors, size_t dim);

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

}


#endif // TENSOR_H
