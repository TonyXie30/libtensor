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
#include <memory>

namespace ts
{
    struct Slice
    {
        size_t start;
        size_t end; // 不包含

        Slice(size_t s, size_t e) : start(s), end(e) {}
        Slice(std::vector<size_t> se)
        {
            if (se.size() != 2)
            {
                throw std::invalid_argument("The number of parameters is incorrect. (When creating slices)");
            }
            else
            {
                start = se[0];
                end = se[1];
            }
        }
        Slice(size_t s) : start(s), end(s + 1) {}
    };
    // Tensor模板类定义
    template <typename T>
    class Tensor
    {
        static_assert(std::is_same<T, int>::value || std::is_same<T, double>::value,
                      "Tensor can only be of type int or double."); // 限制类型为int或double

    public:
        // 默认构造函数
        Tensor() : Tensor(std::vector<T>(1, 0), {1}) {}
        Tensor(const std::vector<T> &data, const std::vector<size_t> &shape);
        Tensor(std::shared_ptr<std::vector<T>> data,
               const std::vector<size_t> &shape,
               const std::vector<Slice> &slices,
               const std::vector<size_t> &strides);
        Tensor(std::shared_ptr<std::vector<T>> data,
               const std::vector<size_t> &shape,
               const std::vector<Slice> &slices,
               const std::vector<size_t> &strides,
               const int i);

        // 获取Tensor形状
        std::vector<size_t> get_shape() const;

        // 访问元素（非const版本）tensor({1,2,3...})
        T &operator()(const std::vector<size_t> &indexes);

        // 访问元素（const版本）tensor({1,2,3...})
        const T &operator()(const std::vector<size_t> &indexes) const;

        Tensor<T> operator()(const size_t s);

        Tensor<T> operator()(size_t s, const std::vector<size_t> &site);

        Tensor<T> &operator=(const T &value);

        Tensor<T> &operator=(const std::vector<T> &data);

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

        Tensor<T> slice(const std::vector<Slice> &slices) const;

        // 连接tensor
        template <typename U>
        friend Tensor<U> cat(const std::vector<Tensor<U>> &tensors, size_t dim);

        // 重复tensor
        template <typename U>
        friend Tensor<U> tile(const Tensor<U> &tensor, const std::vector<size_t> &dims);

        // 递归连接高维张量的辅助函数
        template <typename U>
        friend void recursiveCat(const Tensor<U> &input, Tensor<U> &output, size_t dim, std::vector<size_t> &indexes, size_t current_dim, size_t start_index);
        // 静态成员函数转置
        static Tensor<T> transpose(const Tensor<T> &tensor, size_t dim1, size_t dim2);

        // 成员函数转置
        Tensor<T> transpose(size_t dim1, size_t dim2) const;

        // 视图函数的成员函数版本
        Tensor<T> view(const std::vector<size_t> &new_shape);

        // 静态成员函数版本
        static Tensor<T> view(const Tensor<T> &tensor, const std::vector<size_t> &new_shape);

        void setAllValuesRecursive(Tensor<T> &tensor, const T &value, std::vector<size_t> &indices, size_t dim);

        void setLastValuesRecursive(Tensor<T> &tensor, const std::vector<T> &data, std::vector<size_t> &indices, size_t dim);
        // add
        Tensor<T> operator+(Tensor<T> adder);

        Tensor<T> add(Tensor<T> adder);

        Tensor<T> add(double value);

        // sub
        Tensor<T> operator-(Tensor<T> subtractor);

        Tensor<T> sub(Tensor<T> subtractor);

        Tensor<T> sub(double value);

        Tensor<T> mul(const Tensor<T> &multiplier);

        Tensor<T> operator*(const Tensor<T> &multi);

        Tensor<T> mul(double scala);

        Tensor<T> div(const Tensor<T> &t1, const Tensor<T> &t2);

        Tensor<T> div(double scalar);

        Tensor<T> operator/(const Tensor<T> &dividend);

        Tensor<T> div(const Tensor<T> &tensor);

        // friend void recursive_einsum<T>(
        // const std::vector<Tensor<T>>& tensors,
        // Tensor<T>& result,
        // const std::string& einsum_str,
        // std::vector<int>& current_indices,
        // int depth
        // );

    private:
        std::shared_ptr<std::vector<T>> data_;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;

        std::vector<Slice> slices_;
        bool is_slice_;
        size_t calculate_index(const std::vector<size_t> &indexes) const;

        Tensor<T> sum(int dim) const;

        Tensor<double> mean(int dim);

        Tensor<T> max(int dim);

        Tensor<T> min(int dim);
    };

    template <typename T>
    Tensor<T> cat(const std::vector<Tensor<T>> &tensors, size_t dim)
    {
        if (tensors.empty())
        {
            throw std::invalid_argument("Tensor list is empty.");
        }

        // 检查维度是否有效
        for (const auto &tensor : tensors)
        {
            if (dim >= tensor.get_shape().size())
            {
                throw std::invalid_argument("Invalid dimension for concatenation.");
            }
        }

        // 检查连接的张量维度是否一致
        // for (size_t i = 1; i < tensors.size(); ++i)
        // {
        //     if (tensors[i].get_shape()[dim] != tensors[0].get_shape()[dim])
        //     {
        //         throw std::invalid_argument("Invalid tensor shapes for concatenation.");
        //     }
        // }

        // 计算新张量的形状
        std::vector<size_t> new_shape = tensors[0].get_shape();
        size_t total_dim_size = 0;
        for (const auto &tensor : tensors)
        {
            total_dim_size += tensor.get_shape()[dim];
        }
        new_shape[dim] = total_dim_size;

        // 创建新张量
        Tensor<T> result = Tensor<T>::zeros(new_shape);

        // 进行连接操作
        std::vector<size_t> indexes(new_shape.size(), 0);
        size_t start_index = 0;
        for (const auto &tensor : tensors)
        {
            recursiveCat(tensor, result, dim, indexes, 0, start_index);
            start_index += tensor.get_shape()[dim];
        }

        return result;
    }

    template <typename T>
    void recursiveCat(const Tensor<T> &input, Tensor<T> &output, size_t dim, std::vector<size_t> &indexes, size_t current_dim, size_t start_index)
    {
        if (current_dim == dim)
        {
            for (size_t i = 0; i < input.get_shape()[dim]; ++i)
            {
                indexes[dim] = start_index + i;
                recursiveCat(input, output, dim, indexes, current_dim + 1, start_index);
            }
        }
        else if (current_dim < output.get_shape().size())
        {
            for (size_t i = 0; i < input.get_shape()[current_dim]; ++i)
            {
                indexes[current_dim] = i;
                recursiveCat(input, output, dim, indexes, current_dim + 1, start_index);
            }
        }
        else
        {
            auto input_index = indexes;
            input_index[dim] -= start_index; // 确保 output_index[dim] 不超过 input 的 dim 维度大小
            output(indexes) = input(input_index);
        }
    }

    template <typename T>
    Tensor<T> tile(const Tensor<T> &tensor, const std::vector<size_t> &dims)
    {
        // if (dims.size() != tensor.get_shape().size())
        // {
        //     throw std::invalid_argument("Dimensions for tiling do not match the tensor shape.");
        // }

        Tensor<int> result = tensor;
        size_t count = 0;
        Tensor<int> my_tensor = tensor;
        for (size_t dim : dims)
        {
            for (size_t i = 1; i < dim; i++)
            {
                result = ts::cat<int>({result, my_tensor}, count);
            }
            count++;
            my_tensor = result;
        }
        return result;
    }
}

#endif // TENSOR_H
