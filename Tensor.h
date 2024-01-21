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

        template <typename U>
        friend U *get_data_address(const Tensor<U> &tensor);

        // 获取Tensor形状
        std::vector<size_t> get_shape() const;

        // 获取data
        const std::shared_ptr<std::vector<T>> &getData() const {
            return data_;
        }

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

        // 成员函数转置
        Tensor<T> transpose(size_t dim1, size_t dim2);

        Tensor<T> permute(const std::vector<size_t> &dims);

        static Tensor<T> permute(const Tensor<T> &tensor, const std::vector<size_t> &dims);

        void recursivePermute(size_t dim, const std::vector<size_t> &dims,
                              std::vector<size_t> &indexes, std::vector<size_t> &permuted_indexes,
                              const Tensor<T> source, Tensor<T> &destination);

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

        Tensor<T> div(double scalar);

        Tensor<T> operator/(const Tensor<T> &dividend);

        Tensor<T> div(const Tensor<T> &tensor);

        Tensor<double> sum(int dim) const;

        Tensor<double> mean(int dim) const;

        Tensor<double> max(int dim) const;

        Tensor<double> min(int dim) const;

        Tensor<int> eq(Tensor<T> &tensor);

        Tensor<int> operator==(Tensor<T> &tensor);

        Tensor<int> ne(Tensor<T> &tensor);

        Tensor<int> operator!=(Tensor<T> &tensor);

        Tensor<int> lt(Tensor<T> &tensor);

        Tensor<int> operator<(Tensor<T> &tensor);

        Tensor<int> le(Tensor<T> &tensor);

        Tensor<int> operator<=(Tensor<T> &tensor);

        Tensor<int> gt(Tensor<T> &tensor);

        Tensor<int> operator>(Tensor<T> &tensor);

        Tensor<int> ge(Tensor<T> &tensor);

        Tensor<int> operator>=(Tensor<T> &tensor);

        bool isBool() const {
            return is_bool;
        }

        void setIsBool(bool isBool) {
            is_bool = isBool;
        }

    private:
        std::shared_ptr<std::vector<T>> data_;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;

        std::vector<Slice> slices_;
        bool is_slice_;
        bool is_bool = false;

        size_t calculate_index(const std::vector<size_t> &indexes) const;

    };

    //static Math operations declaration
    template<typename T>
    Tensor<T> add(Tensor<T> a1, Tensor<T> a2) {
        return a1.add(a2);
    }

    // add with mixed types
    template<typename T>
    Tensor<T> add(Tensor<T> t, double value) {
        return t.add(value);
    }

    template<typename T>
    Tensor<T> sub(Tensor<T> a1, Tensor<T> a2) {
        return a1.sub(a2);
    }

    template<typename T>
    Tensor<T> sub(Tensor<T> t, double value) {
        return t.sub(value);
    }

    template<typename T>
    Tensor<T> mul(const Tensor<T> &t1, const Tensor<T> &t2) {
        return t1.mul(t2);
    }

    template<typename T>
    Tensor<T> mul(Tensor<T> &tensor, double scala) {
        return tensor.mul(scala);
    }

    // Divide operation for high-dimensional tensor division
    template<typename T>
    Tensor<T> div(const Tensor<T> &t1, const Tensor<T> &t2) {
        // Check if dimensions are compatible for tensor division
        return t1.div(t2);
    }

    // Divide operation for scalar division
    template<typename T>
    Tensor<T> div(Tensor<T> &tensor, double scalar) {
        return tensor.div(scalar);
    }

    template<typename T>
    Tensor<double> log(const Tensor<T> &tensor) {
        // Create the result Tensor
        std::vector<double> result_data;
        result_data.reserve(tensor.getData()->size());
        // Perform element-wise logarithm
        for (size_t i = 0; i < tensor.getData()->size(); ++i) {
            result_data.push_back(std::log((*tensor.getData())[i]));
        }

        return Tensor<T>(result_data, tensor.get_shape());
    }

    template<typename T>
    Tensor<double> sum(const Tensor<T> &tensor, int dim) {
        return tensor.sum(dim);
    }

    template<typename T>
    Tensor<double> mean(const Tensor<T> &tensor, int dim) {
        return tensor.mean(dim);
    }

    template<typename T>
    Tensor<double> max(const Tensor<T> &tensor, int dim) {
        return tensor.max(dim);
    }

    template<typename T>
    Tensor<double> min(const Tensor<T> &tensor, int dim) {
        return tensor.min(dim);
    }

    template <typename T>
    Tensor<int> eq(Tensor<T> &t1,Tensor<T> &t2){
        return t1.eq(t2);
    }

    template <typename T>
    Tensor<int> ne(Tensor<T> &t1,Tensor<T> &t2){
        return t1.ne(t2);
    }

    template<typename T>
    T *get_data_address(const Tensor<T> &tensor)
    {
        return tensor.data_->data();
    }

    template <typename T>
    std::pair<int,int> recursiveSum(const std::vector<T> &data,const std::vector<size_t> &shape,int start,int end,int target_dim,int current_dim,std::vector<double> &output){
        if (target_dim+1==current_dim){
            std::pair<int,int> ans ;
            ans.first = start;
            ans.second = end;
            return ans;
        }
        else{
            size_t divided_part = shape[current_dim];
            size_t stride = (end - start + 1) / divided_part; //  每一份的步长
            std::vector<std::pair<int,int>> store;
            for (int i = 0; i < divided_part; ++i) {
                std::pair<int,int> ans = recursiveSum(data,shape,start+i*stride,
                                                      start+i*stride+stride-1,target_dim,current_dim+1,output);
                if (ans.first!=-1&&ans.second!=-1){
                    store.push_back(ans);
                }
            }

            if (target_dim==current_dim){
                int single_length = store[0].second - store[0].first + 1;
                for (int i = 0; i < single_length; ++i) {
                    double sum = 0;
                    for (auto & j : store) {
                        sum += data[j.first+i];
                    }
                    output.push_back(sum);
                }
            }

            return std::pair<int,int>(-1,-1);
        }
    }

    template <typename T>
    std::pair<int,int> recursiveMean(const std::vector<T> &data,const std::vector<size_t> &shape,int start,int end,int target_dim,int current_dim,std::vector<double> &output){
        if (target_dim+1==current_dim){
            std::pair<int,int> ans ;
            ans.first = start;
            ans.second = end;
            return ans;
        }
        else{
            size_t divided_part = shape[current_dim];
            size_t stride = (end - start + 1) / divided_part; //  每一份的步长
            std::vector<std::pair<int,int>> store;
            for (int i = 0; i < divided_part; ++i) {
                std::pair<int,int> ans = recursiveMean(data,shape,start+i*stride,
                                                      start+i*stride+stride-1,target_dim,current_dim+1,output);
                if (ans.first!=-1&&ans.second!=-1){
                    store.push_back(ans);
                }
            }

            if (target_dim==current_dim){
                int single_length = store[0].second - store[0].first + 1;
                for (int i = 0; i < single_length; ++i) {
                    double sum = 0;
                    for (auto & j : store) {
                        sum += data[j.first+i];
                    }
                    output.push_back(sum/(double)store.size());
                }
            }

            return std::pair<int,int>(-1,-1);
        }
    }

    template <typename T>
    std::pair<int,int> recursiveMin(const std::vector<T> &data,const std::vector<size_t> &shape,int start,int end,int target_dim,int current_dim,std::vector<double> &output){
        if (target_dim+1==current_dim){
            std::pair<int,int> ans ;
            ans.first = start;
            ans.second = end;
            return ans;
        }
        else{
            size_t divided_part = shape[current_dim];
            size_t stride = (end - start + 1) / divided_part; //  每一份的步长
            std::vector<std::pair<int,int>> store;
            for (int i = 0; i < divided_part; ++i) {
                std::pair<int,int> ans = recursiveMin(data,shape,start+i*stride,
                                                      start+i*stride+stride-1,target_dim,current_dim+1,output);
                if (ans.first!=-1&&ans.second!=-1){
                    store.push_back(ans);
                }
            }

            if (target_dim==current_dim){
                int single_length = store[0].second - store[0].first + 1;
                for (int i = 0; i < single_length; ++i) {
                    double min = std::numeric_limits<double>::infinity();
                    for (auto & j : store) {
                        min = data[j.first+i]<min?data[j.first+i]:min;
                    }
                    output.push_back(min);
                }
            }

            return std::pair<int,int>(-1,-1);
        }
    }

    template <typename T>
    std::pair<int,int> recursiveMax(const std::vector<T> &data,const std::vector<size_t> &shape,int start,int end,int target_dim,int current_dim,std::vector<double> &output){
        if (target_dim+1==current_dim){
            std::pair<int,int> ans ;
            ans.first = start;
            ans.second = end;
            return ans;
        }
        else{
            size_t divided_part = shape[current_dim];
            size_t stride = (end - start + 1) / divided_part; //  每一份的步长
            std::vector<std::pair<int,int>> store;
            for (int i = 0; i < divided_part; ++i) {
                std::pair<int,int> ans = recursiveMax(data,shape,start+i*stride,
                                                      start+i*stride+stride-1,target_dim,current_dim+1,output);
                if (ans.first!=-1&&ans.second!=-1){
                    store.push_back(ans);
                }
            }

            if (target_dim==current_dim){
                int single_length = store[0].second - store[0].first + 1;
                for (int i = 0; i < single_length; ++i) {
                    double max = std::numeric_limits<double>::lowest();
                    for (auto & j : store) {
                        max = data[j.first+i]>max?data[j.first+i]:max;
                    }
                    output.push_back(max);
                }
            }

            return std::pair<int,int>(-1,-1);
        }
    }

    template<typename T>
    Tensor<T> cat(const std::vector<Tensor<T>> &tensors, size_t dim) {
        if (tensors.empty()) {
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

    template <typename T>
    Tensor<T> Tensor<T>::permute(const Tensor<T> &tensor, const std::vector<size_t> &dims)
    {
        Tensor<T> new_tensor(tensor.data_, tensor.shape_, tensor.slices_, tensor.strides_, 1);
        return new_tensor.permute(dims);
    }

    template <typename T>
    Tensor<T> Tensor<T>::permute(const std::vector<size_t> &dims)
    {
        std::vector<size_t> new_shape;
        Tensor<T> source(this->data_, this->shape_, this->slices_, this->strides_, 1);
        for (size_t dim : dims)
        {
            new_shape.push_back(this->shape_[dim]); // 根据dims中的顺序获取新的形状
        }

        Tensor<T> permuted_tensor = view(new_shape); // 根据新的形状创建一个新张量

        std::vector<size_t> indexes(permuted_tensor.get_shape().size(), 0);
        std::vector<size_t> permuted_indexes(permuted_tensor.get_shape().size(), 0);

        // 遍历原始张量，将元素复制到新张量的对应位置
        recursivePermute(0, dims, indexes, permuted_indexes, source, permuted_tensor);
        this->data_ = permuted_tensor.data_;

        return permuted_tensor;
    }

    template <typename T>
    void Tensor<T>::recursivePermute(size_t dim, const std::vector<size_t> &dims,
                                     std::vector<size_t> &indexes, std::vector<size_t> &permuted_indexes,
                                     const Tensor<T> source, Tensor<T> &destination)
    {
        if (dim == source.get_shape().size())
        {
            // 递归结束条件：已经遍历完所有维度
            for (size_t i = 0; i < dims.size(); ++i)
            {
                permuted_indexes[i] = indexes[dims[i]]; // 根据dims中的顺序将索引重新排列
            }
            destination(permuted_indexes) = source(indexes); // 复制元素到新张量的对应位置
        }
        else
        {
            // 对于当前维度，遍历所有可能的索引值
            for (size_t i = 0; i < source.get_shape()[dim]; ++i)
            {
                indexes[dim] = i;
                recursivePermute(dim + 1, dims, indexes, permuted_indexes, source, destination);
            }
        }
    }
}

#endif // TENSOR_H
