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
#include <algorithm>
#include <fstream>
#include <map>
#include <string>
#include <initializer_list>

namespace ts {
    struct Slice {
        size_t start;
        size_t end; // 不包含

        Slice(size_t s, size_t e) : start(s), end(e) {}

        Slice(std::vector<size_t> se) {
            if (se.size() != 2) {
                throw std::invalid_argument("The number of parameters is incorrect. (When creating slices)");
            } else {
                start = se[0];
                end = se[1];
            }
        }

        Slice(size_t s) : start(s), end(s + 1) {}

        Slice() : start(0), end(0) {}
    };
    // Tensor模板类定义
    template<typename T>
    class Tensor {
        static_assert(std::is_same<T, int>::value || std::is_same<T, double>::value ,
                      "Tensor can only be of type int or double"); // 限制类型为int或double

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

        template<typename U>
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
        template<typename U>
        friend Tensor<U> cat(const std::vector<Tensor<U>> &tensors, size_t dim);

        // 重复tensor
        template<typename U>
        friend Tensor<U> tile(const Tensor<U> &tensor, const std::vector<size_t> &dims);

        // 递归连接高维张量的辅助函数
        template<typename U>
        friend void recursiveCat(const Tensor<U> &input, Tensor<U> &output, size_t dim, std::vector<size_t> &indexes,
                                 size_t current_dim, size_t start_index);

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

        void
        setLastValuesRecursive(Tensor<T> &tensor, const std::vector<T> &data, std::vector<size_t> &indices, size_t dim);

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

        Tensor<double> div(double scalar);

        Tensor<double> operator/(const Tensor<T> &dividend);

        Tensor<double> div(const Tensor<T> &tensor);

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

        template<typename U>
        friend void saveTensorToFile(const Tensor<U> &tensor, const std::string &filename);

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
    Tensor<double> div(const Tensor<T> &t1, const Tensor<T> &t2) {
        // Check if dimensions are compatible for tensor division
        return t1.div(t2);
    }

    // Divide operation for scalar division
    template<typename T>
    Tensor<double> div(Tensor<T> &tensor, double scalar) {
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

        return Tensor<double>(result_data, tensor.get_shape());
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

    template<typename T>
    Tensor<int> eq(Tensor<T> &t1, Tensor<T> &t2) {
        return t1.eq(t2);
    }

    template<typename T>
    Tensor<int> ne(Tensor<T> &t1, Tensor<T> &t2) {
        return t1.ne(t2);
    }

    template<typename T>
    Tensor<T> save(Tensor<T> tensor, std::string filename);

    template<typename T>
    Tensor<T> load(std::string filename);

    template<typename T>
    T *get_data_address(const Tensor<T> &tensor) {
        return tensor.data_->data();
    }

    template<typename T>
    std::pair<int, int>
    recursiveSum(const std::vector<T> &data, const std::vector<size_t> &shape, int start, int end, int target_dim,
                 int current_dim, std::vector<double> &output) {
        if (target_dim + 1 == current_dim) {
            std::pair<int, int> ans;
            ans.first = start;
            ans.second = end;
            return ans;
        } else {
            size_t divided_part = shape[current_dim];
            size_t stride = (end - start + 1) / divided_part; //  每一份的步长
            std::vector<std::pair<int, int>> store;
            for (int i = 0; i < divided_part; ++i) {
                std::pair<int, int> ans = recursiveSum(data, shape, start + i * stride,
                                                       start + i * stride + stride - 1, target_dim, current_dim + 1,
                                                       output);
                if (ans.first != -1 && ans.second != -1) {
                    store.push_back(ans);
                }
            }

            if (target_dim == current_dim) {
                int single_length = store[0].second - store[0].first + 1;
                for (int i = 0; i < single_length; ++i) {
                    double sum = 0;
                    for (auto &j: store) {
                        sum += data[j.first + i];
                    }
                    output.push_back(sum);
                }
            }

            return std::pair<int, int>(-1, -1);
        }
    }

    template<typename T>
    std::pair<int, int>
    recursiveMean(const std::vector<T> &data, const std::vector<size_t> &shape, int start, int end, int target_dim,
                  int current_dim, std::vector<double> &output) {
        if (target_dim + 1 == current_dim) {
            std::pair<int, int> ans;
            ans.first = start;
            ans.second = end;
            return ans;
        } else {
            size_t divided_part = shape[current_dim];
            size_t stride = (end - start + 1) / divided_part; //  每一份的步长
            std::vector<std::pair<int, int>> store;
            for (int i = 0; i < divided_part; ++i) {
                std::pair<int, int> ans = recursiveMean(data, shape, start + i * stride,
                                                        start + i * stride + stride - 1, target_dim, current_dim + 1,
                                                        output);
                if (ans.first != -1 && ans.second != -1) {
                    store.push_back(ans);
                }
            }

            if (target_dim == current_dim) {
                int single_length = store[0].second - store[0].first + 1;
                for (int i = 0; i < single_length; ++i) {
                    double sum = 0;
                    for (auto &j: store) {
                        sum += data[j.first + i];
                    }
                    output.push_back(sum / (double) store.size());
                }
            }

            return std::pair<int, int>(-1, -1);
        }
    }

    template<typename T>
    std::pair<int, int>
    recursiveMin(const std::vector<T> &data, const std::vector<size_t> &shape, int start, int end, int target_dim,
                 int current_dim, std::vector<double> &output) {
        if (target_dim + 1 == current_dim) {
            std::pair<int, int> ans;
            ans.first = start;
            ans.second = end;
            return ans;
        } else {
            size_t divided_part = shape[current_dim];
            size_t stride = (end - start + 1) / divided_part; //  每一份的步长
            std::vector<std::pair<int, int>> store;
            for (int i = 0; i < divided_part; ++i) {
                std::pair<int, int> ans = recursiveMin(data, shape, start + i * stride,
                                                       start + i * stride + stride - 1, target_dim, current_dim + 1,
                                                       output);
                if (ans.first != -1 && ans.second != -1) {
                    store.push_back(ans);
                }
            }

            if (target_dim == current_dim) {
                int single_length = store[0].second - store[0].first + 1;
                for (int i = 0; i < single_length; ++i) {
                    double min = std::numeric_limits<double>::infinity();
                    for (auto &j: store) {
                        min = data[j.first + i] < min ? data[j.first + i] : min;
                    }
                    output.push_back(min);
                }
            }

            return std::pair<int, int>(-1, -1);
        }
    }

    template<typename T>
    std::pair<int, int>
    recursiveMax(const std::vector<T> &data, const std::vector<size_t> &shape, int start, int end, int target_dim,
                 int current_dim, std::vector<double> &output) {
        if (target_dim + 1 == current_dim) {
            std::pair<int, int> ans;
            ans.first = start;
            ans.second = end;
            return ans;
        } else {
            size_t divided_part = shape[current_dim];
            size_t stride = (end - start + 1) / divided_part; //  每一份的步长
            std::vector<std::pair<int, int>> store;
            for (int i = 0; i < divided_part; ++i) {
                std::pair<int, int> ans = recursiveMax(data, shape, start + i * stride,
                                                       start + i * stride + stride - 1, target_dim, current_dim + 1,
                                                       output);
                if (ans.first != -1 && ans.second != -1) {
                    store.push_back(ans);
                }
            }

            if (target_dim == current_dim) {
                int single_length = store[0].second - store[0].first + 1;
                for (int i = 0; i < single_length; ++i) {
                    double max = std::numeric_limits<double>::lowest();
                    for (auto &j: store) {
                        max = data[j.first + i] > max ? data[j.first + i] : max;
                    }
                    output.push_back(max);
                }
            }

            return std::pair<int, int>(-1, -1);
        }
    }

    template<typename T>
    Tensor<T> cat(const std::vector<Tensor<T>> &tensors, size_t dim) {
        if (tensors.empty()) {
            throw std::invalid_argument("Tensor list is empty.");
        }

        // 检查维度是否有效
        for (const auto &tensor: tensors) {
            if (dim >= tensor.get_shape().size()) {
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
        for (const auto &tensor: tensors) {
            total_dim_size += tensor.get_shape()[dim];
        }
        new_shape[dim] = total_dim_size;

        // 创建新张量
        Tensor<T> result = Tensor<T>::zeros(new_shape);

        // 进行连接操作
        std::vector<size_t> indexes(new_shape.size(), 0);
        size_t start_index = 0;
        for (const auto &tensor: tensors) {
            recursiveCat(tensor, result, dim, indexes, 0, start_index);
            start_index += tensor.get_shape()[dim];
        }

        return result;
    }

    template<typename T>
    void recursiveCat(const Tensor<T> &input, Tensor<T> &output, size_t dim, std::vector<size_t> &indexes,
                      size_t current_dim, size_t start_index) {
        if (current_dim == dim) {
            for (size_t i = 0; i < input.get_shape()[dim]; ++i) {
                indexes[dim] = start_index + i;
                recursiveCat(input, output, dim, indexes, current_dim + 1, start_index);
            }
        } else if (current_dim < output.get_shape().size()) {
            for (size_t i = 0; i < input.get_shape()[current_dim]; ++i) {
                indexes[current_dim] = i;
                recursiveCat(input, output, dim, indexes, current_dim + 1, start_index);
            }
        } else {
            auto input_index = indexes;
            input_index[dim] -= start_index; // 确保 output_index[dim] 不超过 input 的 dim 维度大小
            output(indexes) = input(input_index);
        }
    }

    template<typename T>
    Tensor<T> tile(const Tensor<T> &tensor, const std::vector<size_t> &dims) {
        Tensor<int> result = tensor;
        size_t count = 0;
        Tensor<int> my_tensor = tensor;
        for (size_t dim: dims) {
            for (size_t i = 1; i < dim; i++) {
                result = ts::cat<int>({result, my_tensor}, count);
            }
            count++;
            my_tensor = result;
        }
        return result;
    }

    template<typename T>
    Tensor<T> Tensor<T>::permute(const Tensor<T> &tensor, const std::vector<size_t> &dims) {
        Tensor<T> new_tensor(tensor.data_, tensor.shape_, tensor.slices_, tensor.strides_, 1);
        return new_tensor.permute(dims);
    }

    template<typename T>
    Tensor<T> Tensor<T>::permute(const std::vector<size_t> &dims) {
        std::vector<size_t> new_shape;
        Tensor<T> source(this->data_, this->shape_, this->slices_, this->strides_, 1);
        for (size_t dim: dims) {
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

    template<typename T>
    void Tensor<T>::recursivePermute(size_t dim, const std::vector<size_t> &dims,
                                     std::vector<size_t> &indexes, std::vector<size_t> &permuted_indexes,
                                     const Tensor<T> source, Tensor<T> &destination) {
        if (dim == source.get_shape().size()) {
            // 递归结束条件：已经遍历完所有维度
            for (size_t i = 0; i < dims.size(); ++i) {
                permuted_indexes[i] = indexes[dims[i]]; // 根据dims中的顺序将索引重新排列
            }
            destination(permuted_indexes) = source(indexes); // 复制元素到新张量的对应位置
        } else {
            // 对于当前维度，遍历所有可能的索引值
            for (size_t i = 0; i < source.get_shape()[dim]; ++i) {
                indexes[dim] = i;
                recursivePermute(dim + 1, dims, indexes, permuted_indexes, source, destination);
            }
        }
    }

    template<typename T>
    void saveTensorToFile(const Tensor<T> &tensor, const std::string &filename) {
        std::ofstream file(filename, std::ios::binary);
        if (file.is_open()) {
            // 写入形状
            size_t shapeSize = tensor.get_shape().size();
            file.write(reinterpret_cast<const char *>(&shapeSize), sizeof(size_t));
            file.write(reinterpret_cast<const char *>(tensor.get_shape().data()), shapeSize * sizeof(size_t));

            // 写入切片信息
            size_t sliceSize = tensor.slices_.size();
            file.write(reinterpret_cast<const char *>(&sliceSize), sizeof(size_t));
            for (const auto &s: tensor.slices_) {
                file.write(reinterpret_cast<const char *>(&s.start), sizeof(size_t));
                file.write(reinterpret_cast<const char *>(&s.end), sizeof(size_t));
            }

            // 写入步长信息
            size_t strideSize = tensor.strides_.size();
            file.write(reinterpret_cast<const char *>(&strideSize), sizeof(size_t));
            std::cout << std::endl;
            file.write(reinterpret_cast<const char *>(tensor.strides_.data()), strideSize * sizeof(size_t));

            // 写入数据
            file.write(reinterpret_cast<const char *>(tensor.data_->data()), tensor.data_->size() * sizeof(T));
            file.close();
            std::cout << "Tensor saved to " << filename << std::endl;
        } else {
            std::cerr << "Error: Unable to open file for writing" << std::endl;
        }
    }

    template<typename T>
    Tensor<T> loadTensorFromFile(const std::string &filename) {
        std::ifstream file(filename, std::ios::binary);
        if (file.is_open()) {
            size_t shapeSize;
            file.read(reinterpret_cast<char *>(&shapeSize), sizeof(size_t));
            std::vector<size_t> shape(shapeSize);
            file.read(reinterpret_cast<char *>(shape.data()), shapeSize * sizeof(size_t));

            // 读取切片信息
            size_t sliceSize;
            file.read(reinterpret_cast<char *>(&sliceSize), sizeof(size_t));
            std::vector<Slice> slices(sliceSize);
            for (size_t i = 0; i < sliceSize; ++i) {
                size_t start, end;
                file.read(reinterpret_cast<char *>(&start), sizeof(size_t));
                file.read(reinterpret_cast<char *>(&end), sizeof(size_t));
                slices[i] = Slice(start, end);
            }

            // 读取步长信息
            size_t strideSize;
            file.read(reinterpret_cast<char *>(&strideSize), sizeof(size_t));
            std::vector<size_t> strides(strideSize);
            file.read(reinterpret_cast<char *>(strides.data()), strideSize * sizeof(size_t));

            size_t dataSize = 1;
            for (size_t s: shape) {
                dataSize *= s;
            }
            std::vector<T> data(dataSize);
            std::shared_ptr<std::vector<T>> data_ = std::make_shared<std::vector<T>>(dataSize);
            file.read(reinterpret_cast<char *>(data_->data()), dataSize * sizeof(T));
            std::cout << std::endl;
            Tensor<T> tensor(data_, shape, slices, strides, 1);

            file.close();
            std::cout << "Tensor loaded from " << filename << std::endl;

            return tensor;
        } else {
            std::cerr << "Error: Unable to open file for reading" << std::endl;
            return Tensor<T>();
        }
    }

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

    inline bool isBroadcastable(std::vector<size_t> t1_shape,std::vector<size_t> &t2_shape) {
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
    Tensor<T> broadcast(const Tensor<T> &input, std::vector<size_t> &other) {
        // Check if input shape is compatible with target shape for broadcasting
        if (!isBroadcastable(input.get_shape(), other)) {
            throw std::invalid_argument("Cannot broadcast tensors with incompatible shapes.");
        }

        std::vector<size_t> origin = input.get_shape();
        std::vector<size_t> multi_vector;

        std::reverse(origin.begin(), origin.end());
        std::reverse(other.begin(), other.end());

        for (size_t i = 0; i < other.size(); ++i) {
            size_t dim = i > origin.size() - 1 ? 1 : origin[i];
            size_t result = dim == other[i]? 1:other[i];
            multi_vector.push_back(result);
        }

        std::reverse(multi_vector.begin(),multi_vector.end());
        return tile(input, multi_vector);
    }


    template<typename T>
    void handle_ellipsis(
            std::string& str,
            const std::vector<Tensor<T>>& tensors,
            std::vector<char>& indices_order,
            bool is_input
    ) {
        size_t ellipsis_pos = str.find("...");
        if (ellipsis_pos != std::string::npos) {
            size_t tensor_dim = tensors[0].get_shape().size();
            size_t explicit_dims = std::count_if(str.begin(), str.end(), [](char c) { return std::isalpha(c); });
            size_t omitted_dims = tensor_dim - explicit_dims;

            std::string new_indices;
            char new_index_char = 'a';
            for (size_t i = 0; i < omitted_dims; ++i) {
                while (str.find(new_index_char) != std::string::npos) {
                    new_index_char++;
                }
                new_indices += new_index_char++;
            }

            str.replace(ellipsis_pos, 3, new_indices);
            if (is_input) {
                indices_order.insert(indices_order.end(), new_indices.begin(), new_indices.end());
            }
        }
    }
    template<typename T>
    void parse_einsum_str(
            const std::string& einsum_str,
            const std::vector<Tensor<T>>& tensors,
            std::map<char, std::map<int, int>>& index_dim_map, // 输入张量维度映射
            std::map<char, int>& output_index_dim_map,         // 输出张量维度映射
            std::vector<char>& indices_order                   // 索引字符的顺序
    ) {
        index_dim_map.clear();
        output_index_dim_map.clear();
        indices_order.clear();

        auto arrow_pos = einsum_str.find("->");
        auto input_str = einsum_str.substr(0, arrow_pos);
        // 确保 "->" 后面有字符
        auto output_str = (arrow_pos != std::string::npos && arrow_pos + 2 < einsum_str.length())
                          ? einsum_str.substr(arrow_pos + 2)
                          : "";
        handle_ellipsis(input_str, tensors, indices_order, true);

        // 处理输出字符串中的省略号
        handle_ellipsis(output_str, tensors, indices_order, false);
        int tensor_index = 0;
        int dim_index = 0;
        for (char ch : input_str) {
            if (ch == ',') {
                tensor_index++;
                dim_index = 0;
                continue;
            }
            if (ch != ' ') {
                index_dim_map[ch][tensor_index] = dim_index++;
                if (std::find(indices_order.begin(), indices_order.end(), ch) == indices_order.end()) {
                    indices_order.push_back(ch);
                }
            }
        }

        // 解析输出字符串，记录输出张量中每个索引的维度位置
        if (!output_str.empty()) {
            dim_index = 0;
            for (char ch : output_str) {
                if (ch != ' ') {
                    output_index_dim_map[ch] = dim_index++;
                }
            }
        }
    }

//检查各个数组对应维度是否相同，依据同一字母代表的各个张量的维度来比较。
    template<typename T>
    bool check_dimension_consistency(
            const std::vector<Tensor<T>>& tensors,
            const std::map<char, std::map<int, int>>& index_dim_map
    ) {
        for (const auto& [index, dim_map] : index_dim_map) {
            int last_dim_size = -1;
            for (const auto& [tensor_index, dim_index] : dim_map) {
                if (tensor_index >= tensors.size()) {
                    throw std::runtime_error("Tensor index out of range.");
                }
                const auto& shape = tensors[tensor_index].get_shape();
                if (dim_index >= shape.size()) {
                    throw std::runtime_error("Dimension index out of range.");
                }
                int dim_size = shape[dim_index];
                if (last_dim_size != -1 && dim_size != last_dim_size) {
                    return false; // 维度长度不一致
                }
                last_dim_size = dim_size;
            }
        }
        return true; // 所有共享索引的维度长度一致
    }

    template<typename T>
    Tensor<T> create_output_tensor(
            const std::vector<Tensor<T>>& tensors,
            const std::vector<char>& indices_order,
            const std::vector<int>& dim_lengths,
            const std::map<char, int>& output_index_dim_map
    ) {
        if (output_index_dim_map.empty()) {
            // 输出是一个标量，创建一个包含单个元素的一维张量
            return Tensor<T>::zeros({1});
        }
        std::vector<size_t> shape(output_index_dim_map.size(), 0); // 初始化形状向量

        for (const auto& [index, dim_position] : output_index_dim_map) {
            auto it = std::find(indices_order.begin(), indices_order.end(), index);
            if (it != indices_order.end()) {
                int order_position = std::distance(indices_order.begin(), it);
                size_t dim_size = dim_lengths[order_position];
                shape[dim_position] = dim_size; // 设置输出张量的对应维度的大小
            } else {
                throw std::runtime_error("Index not found in indices_order.");
            }
        }
        return Tensor<T>::zeros(shape); // 使用形状创建新的 Tensor
    }

//创建索引数组
    template<typename T>
    std::vector<std::vector<size_t>> generate_tensor_indices_for_each_tensor(
            const std::vector<Tensor<T>>& tensors,
            const std::vector<char>& indices_order,
            const std::vector<int>& current_indices,
            const std::map<char, std::map<int, int>>& index_dim_map
    ) {
        std::vector<std::vector<size_t>> all_tensor_indices;

        for (size_t tensor_idx = 0; tensor_idx < tensors.size(); ++tensor_idx) {
            std::vector<size_t> tensor_indices(tensors[tensor_idx].get_shape().size(), 0); // 初始化为零
            for (size_t i = 0; i < indices_order.size(); ++i) {
                char index_char = indices_order[i];
                if (index_dim_map.find(index_char) != index_dim_map.end() &&
                    index_dim_map.at(index_char).find(tensor_idx) != index_dim_map.at(index_char).end()) {
                    // 获取当前索引字符在当前张量中的维度位置
                    int dim_position = index_dim_map.at(index_char).at(tensor_idx);
                    tensor_indices[dim_position] = current_indices[i]; // 设置对应维度的索引值
                }
            }
            all_tensor_indices.push_back(tensor_indices);
        }

        return all_tensor_indices;
    }
//创建输出索引数组
    template<typename T>
    std::vector<size_t> generate_output_indices(
            const Tensor<T>& output_tensor,
            const std::vector<char>& indices_order,
            const std::vector<int>& current_indices,
            const std::map<char, int>& output_index_dim_map
    ) {
        if (output_index_dim_map.empty()) {
            return std::vector<size_t>{0};
        }
        std::vector<size_t> output_indices(output_tensor.get_shape().size(), 0); // 根据输出张量的维度大小初始化

        for (size_t i = 0; i < indices_order.size(); ++i) {
            char index_char = indices_order[i];
            if (output_index_dim_map.find(index_char) != output_index_dim_map.end()) {
                // 获取当前索引字符在输出张量中的维度位置
                int dim_position = output_index_dim_map.at(index_char);
                if (dim_position < output_indices.size()) {
                    output_indices[dim_position] = current_indices[i]; // 设置对应维度的索引值
                }
            }
        }

        return output_indices;
    }
//获取各个字母对应的维度长度，方便迭代时进行计算。
    template<typename T>
    std::vector<int> generate_dim_lengths(
            const std::vector<Tensor<T>>& tensors,
            const std::vector<char>& indices_order,
            const std::map<char, std::map<int, int>>& index_dim_map
    ) {
        std::vector<int> dim_lengths;

        for (char index_char : indices_order) {
            if (index_dim_map.find(index_char) != index_dim_map.end()) {
                const auto& tensor_dims = index_dim_map.at(index_char);
                auto tensor_index = tensor_dims.begin()->first;    // 第一个出现的张量索引
                auto dim_index = tensor_dims.begin()->second;

                if (tensor_index < tensors.size() && dim_index < tensors[tensor_index].get_shape().size()) {
                    dim_lengths.push_back(tensors[tensor_index].get_shape()[dim_index]);
                } else {
                    throw std::runtime_error("Invalid tensor or dimension index.");
                }
            }
        }

    return dim_lengths;
}
//判断是否要提取对角线元素
inline bool is_diagonal_extraction(const std::string& einsum_str) {
    if (einsum_str.length() == 5 && einsum_str[0] == einsum_str[1] &&
        einsum_str[3] == '>' && einsum_str[4] == einsum_str[0]) {
        return true;
    }
    return false;
}
//提取对角线元素
template<typename T>
Tensor<T> extract_diagonal(const Tensor<T>& tensor) {
    const auto& shape = tensor.get_shape();
    if (shape.size() != 2 || shape[0] != shape[1]) {
        throw std::runtime_error("Tensor must be a square matrix to extract diagonal.");
    }

    Tensor<T> result = Tensor<T>::zeros({shape[0]});
    for (size_t i = 0; i < shape[0]; ++i) {
        result({i}) = tensor({i, i});
    }

    return result;
}
template<typename T>
void check_tensor_count(const std::vector<Tensor<T>>& tensors, const std::string& einsum_str) {
    size_t comma_count = 0;
    for (char ch : einsum_str) {
        if (ch == ',') {
            comma_count++;
        }
    }
    size_t tensor_required = comma_count + 1;  // 逗号数量加 1 是需要的张量数量
    if (tensors.size() != tensor_required) {
        throw std::runtime_error("Number of tensors does not match the number required by the einsum string.");
    }
}

inline void check_index_consistency(const std::string& einsum_str) {
    auto arrow_pos = einsum_str.find("->");
    if (arrow_pos == std::string::npos) {
        throw std::runtime_error("einsum string does not contain '->'.");
    }

    // 创建一个足够大的数组来覆盖所有可能的字符索引
    bool index_present[256] = {false};

    // 标记输入索引
    for (size_t i = 0; i < arrow_pos; ++i) {
        if (isalpha(einsum_str[i])) {
            index_present[static_cast<unsigned char>(einsum_str[i])] = true;
        }
    }

    // 检查输出索引是否在输入索引中
    for (size_t i = arrow_pos + 2; i < einsum_str.length(); ++i) {
        if (isalpha(einsum_str[i]) && !index_present[static_cast<unsigned char>(einsum_str[i])]) {
            throw std::runtime_error("Output indices contain characters not found in input indices.");
        }
    }
}
template<typename T>
void recursive_einsum(
    const std::vector<Tensor<T>>& tensors,
    Tensor<T>& result,
    const std::vector<char>& indices_order,
    std::vector<int>& current_indices,
    const std::map<char, int>& output_index_dim_map,
    const std::map<char, std::map<int, int>>& index_dim_map,
    const std::vector<int>& dim_lengths,
    int depth=0
) {
    if (depth == indices_order.size()) {
        // 到达最内层，执行计算
        auto input_indices = generate_tensor_indices_for_each_tensor(tensors, indices_order, current_indices, index_dim_map);
        auto output_indices = generate_output_indices(result, indices_order, current_indices, output_index_dim_map);

        // T sum = 0;
        // for (size_t i = 0; i < tensors.size(); ++i) {
        //     sum += tensors[i](input_indices[i]); // 累加来自每个张量的值
        // }
        T sum = 1;
        for (size_t i = 0; i < tensors.size(); ++i) {
            sum *= tensors[i](input_indices[i]); // 累乘来自每个张量的值
        }
        result(output_indices) += sum; // 更新结果张量
    } else {
        int dim_size = dim_lengths[depth];
        for (int i = 0; i < dim_size; ++i) {
            current_indices[depth] = i;
            recursive_einsum(tensors, result, indices_order, current_indices, output_index_dim_map, index_dim_map, dim_lengths, depth + 1);
        }
    }
}

template<typename T>
Tensor<T> execute_einsum(
    const std::vector<Tensor<T>>& tensors,
    const std::string& einsum_str
) {
    check_tensor_count(tensors, einsum_str);
    check_index_consistency(einsum_str);
    if (is_diagonal_extraction(einsum_str)) {
        return extract_diagonal<T>(tensors[0]);
    }
    // 解析 einsum 字符串
    std::map<char, std::map<int, int>> index_dim_map;
    std::map<char, int> output_index_dim_map;
    std::vector<char> indices_order;
    parse_einsum_str(einsum_str, tensors , index_dim_map, output_index_dim_map, indices_order);

        // 检查维度一致性
        if (!check_dimension_consistency(tensors, index_dim_map)) {
            throw std::runtime_error("Dimension mismatch among tensors.");
        }

        // 创建输出张量
        std::vector<int> dim_lengths = generate_dim_lengths(tensors, indices_order, index_dim_map);
        Tensor<T> result = create_output_tensor(tensors, indices_order, dim_lengths, output_index_dim_map);

        // 初始化当前索引并执行递归计算
        std::vector<int> current_indices(indices_order.size(), 0);
        recursive_einsum(tensors, result, indices_order, current_indices, output_index_dim_map, index_dim_map, dim_lengths);

        return result;
    }

}

#endif // TENSOR_H
