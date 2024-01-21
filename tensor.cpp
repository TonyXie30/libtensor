#include "Tensor.h"

namespace ts
{
    template <typename T>
    Tensor<T>::Tensor(const std::vector<T> &data, const std::vector<size_t> &shape)
        : data_(std::make_shared<std::vector<T>>(data)), shape_(shape), is_slice_(false)
    {
        // 验证形状是否有效
        if (shape.empty())
        {
            throw std::invalid_argument("Shape cannot be empty.(when create tensor)");
        }

        // 验证每个维度的大小
        for (size_t dim : shape)
        {
            if (dim == 0 || dim > 1000)
            {
                throw std::invalid_argument("Shape dimensions must be greater than zero and should not be too large.(when create tensor)");
            }
        }

        // 计算步长
        strides_.resize(shape_.size());
        size_t stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i)
        {
            strides_[i] = stride;
            stride *= shape_[i];
        }
        slices_.clear();
        // 验证数据大小
        if (data_->size() != stride)
        {
            throw std::logic_error("Data size does not match the tensor shape.(when create tensor)");
        }
    }
    template <typename T>
    Tensor<T>::Tensor(std::shared_ptr<std::vector<T>> data,
                      const std::vector<size_t> &shape,
                      const std::vector<Slice> &slices,
                      const std::vector<size_t> &strides)
        : data_(std::move(data)), shape_(shape), slices_(slices), strides_(strides), is_slice_(true)
    {
        // 检查形状、切片和步长的有效性
        if (shape.size() != slices.size() || shape.size() != strides.size())
        {
            throw std::invalid_argument("Shape, slices, and strides must have the same size.");
        }
    }


    template <typename T>
    Tensor<T>::Tensor(std::shared_ptr<std::vector<T>> data,
                      const std::vector<size_t> &shape,
                      const std::vector<Slice> &slices,
                      const std::vector<size_t> &strides,
                      const int i)
        : shape_(shape), slices_(slices), strides_(strides), is_slice_(false)
    {
        // 深拷贝数据
        data_ = std::make_shared<std::vector<T>>(*data);

        // 检查形状、切片和步长的有效性
        if (shape.size() != strides.size())
        {
            throw std::invalid_argument("Shape and strides must have the same size.");
        }
    }

    // 获取Tensor形状实现
    template <typename T>
    std::vector<size_t> Tensor<T>::get_shape() const
    {
        return shape_;
    }

    // 访问元素实现（非const版本）
    template <typename T>
    T &Tensor<T>::operator()(const std::vector<size_t> &indexes)
    {
        return (*data_)[calculate_index(indexes)];
    }

    // 访问元素实现（const版本）
    template <typename T>
    const T &Tensor<T>::operator()(const std::vector<size_t> &indexes) const
    {
        return (*data_)[calculate_index(indexes)];
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator()(const size_t s)
    {
        return this->slice({s});
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator()(size_t s, const std::vector<size_t> &site)
    {
        return this->slice({s, site});
    }

    template <typename T>
    Tensor<T> &Tensor<T>::operator=(const T &value)
    {
        std::vector<size_t> loc_shape = this->get_shape();
        std::vector<size_t> indices(loc_shape.size(), 0); // 初始化索引向量
        setAllValuesRecursive(*this, value, indices, 0);  // 将张量中的所有元素都设置为 value
        return *this;
    }

    template <typename T>
    Tensor<T> &Tensor<T>::operator=(const std::vector<T> &data)
    {
        size_t total_size = 1;
        for (size_t i = 0; i < shape_.size(); i++)
        {
            total_size *= shape_[i];
        }

        if (data.size() != total_size)
        {
            throw std::invalid_argument("Data size does not match tensor shape");
        }
        std::vector<size_t> loc_shape = this->get_shape();
        std::vector<size_t> indices(loc_shape.size(), 0); // 初始化索引向量
        setLastValuesRecursive(*this, data, indices, 0);  // 将张量中的所有元素都设置为 value
        return *this;
    }

    template <typename T>
    void Tensor<T>::setLastValuesRecursive(Tensor<T> &tensor, const std::vector<T> &data, std::vector<size_t> &indices, size_t dim)
    {

        int count = tensor.get_shape().size();
        if (dim != count - 1)
        {
            indices[dim] = 0;
            setLastValuesRecursive(*this, data, indices, dim + 1);
            return;
        }

        for (size_t i = 0; i < data.size(); i++)
        {
            indices[dim] = i;
            tensor(indices) = data[i];
        }
    }

    template <typename T>
    void Tensor<T>::setAllValuesRecursive(Tensor<T> &tensor, const T &value, std::vector<size_t> &indices, size_t dim)
    {
        if (dim == tensor.get_shape().size())
        {
            // 达到最后一个维度时，将当前索引处的元素设置为 value
            tensor(indices) = value;
        }
        else
        {
            // 递归遍历下一个维度的所有索引
            for (size_t i = 0; i < tensor.get_shape()[dim]; i++)
            {
                indices[dim] = i;
                setAllValuesRecursive(tensor, value, indices, dim + 1);
            }
        }
    }

    // 计算索引实现
    template <typename T>
    size_t Tensor<T>::calculate_index(const std::vector<size_t> &indexes) const
    {
        if (indexes.size() != shape_.size())
        {
            throw std::invalid_argument("Dimension mismatch.(when use index)");
        }
        size_t index = 0;
        for (size_t i = 0; i < indexes.size(); ++i)
        {
            if (is_slice_ && (indexes[i] < 0 || indexes[i] >= slices_[i].end - slices_[i].start))
            {
                throw std::out_of_range("Tensor index out of slice range.(when use index)");
            }
            if (indexes[i] >= shape_[i])
            {
                throw std::out_of_range("Tensor index out of range.(when use index)");
            }
            size_t adjusted_index = is_slice_ ? indexes[i] + slices_[i].start : indexes[i];
            index += strides_[i] * adjusted_index;
        }
        return index;
    }

    // 输出张量结构与元素的实现
    template <typename T>
    void printTensor(const std::vector<T> &data,
                     const std::vector<size_t> &shape,
                     const std::vector<size_t> &strides,
                     const std::vector<Slice> &slices,
                     bool is_slice,
                     bool is_bool,
                     size_t index = 0,
                     size_t dimension = 0)
    {
        if (dimension == shape.size() - 1)
        {
            std::cout << "[";
            for (size_t i = 0; i < shape[dimension]; ++i)
            {
                // 计算实际索引，考虑到切片的起始位置和步长
                size_t actual_index = index;
                if (is_slice)
                {
                    actual_index += (slices[dimension].start + i) * strides[dimension];
                }
                else
                {
                    actual_index += i * strides[dimension];
                }
                // bool值打印，当值为1时且isbool()成立，打印bool
                if ((data[actual_index] == 1 || data[actual_index] == 0) && is_bool) {
                    std::string ans = data[actual_index] == 1 ? "true" : "false";
                    std::cout << ans;
                } else{
                    std::cout << data[actual_index];
                }
                if (i < shape[dimension] - 1)
                    std::cout << ", ";
            }
            std::cout << "]";
        }
        else
        {
            std::cout << "[";
            for (size_t i = 0; i < shape[dimension]; ++i)
            {
                // 计算下一维度的索引
                size_t next_index = index;
                if (is_slice)
                {
                    // 如果是切片，考虑到切片的起始位置和步长
                    next_index += (slices[dimension].start + i) * strides[dimension];
                }
                else
                {
                    next_index += i * strides[dimension];
                }
                printTensor(data, shape, strides, slices, is_slice, is_bool,next_index, dimension + 1);
                if (i < shape[dimension] - 1)
                    std::cout << ", ";
                if (dimension == 0 && i < shape[dimension] - 1)
                    std::cout << std::endl
                              << " ";
            }

            std::cout << "]";
        }
        if (dimension == 0)
            std::cout << std::endl;
    }

    template <typename T>
    void Tensor<T>::print() const
    {
        // 检查是否是切片，如果是，传递切片信息
        printTensor(*data_, shape_, strides_, slices_, is_slice_, is_bool);
    }

    // rand<>,随机初始化tensor
    // double类型实现
    template <>
    Tensor<double> Tensor<double>::rand(const std::vector<size_t> &shape)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Shape cannot be empty.(when using rand to create tensor)");
        }
        for (size_t i = 0; i < shape.size(); i++)
        {
            if (shape[i] == 0 || shape[i] > 1000)
            {
                throw std::invalid_argument(
                    "Shape dimensions must be greater than zero and should not be too large.(when using rand to create tensor)");
            }
        }
        size_t total_number = 1;
        for (size_t i = shape.size(); i-- > 0;)
        {
            total_number *= shape[i];
        }
        std::vector<double> data(total_number);
        std::random_device rd;                                // 随机数生成器种子
        std::mt19937 gen(rd());                               // 基于mt19937算法的随机数生成器
        std::uniform_real_distribution<double> dis(0.0, 1.0); // 定义一个从0.0到1.0的均匀分布

        for (double &val : data)
        {
            val = dis(gen); // 生成一个随机数并赋值
        }
        return Tensor<double>(data, shape);
    }

    // rand<>,随机初始化tensor
    //  int类型的实现
    template <>
    Tensor<int> Tensor<int>::rand(const std::vector<size_t> &shape)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Shape cannot be empty.(when using rand to create tensor)");
        }
        for (size_t i = 0; i < shape.size(); i++)
        {
            if (shape[i] == 0 || shape[i] > 1000)
            {
                throw std::invalid_argument(
                    "Shape dimensions must be greater than zero and should not be too large.(when using rand to create tensor)");
            }
        }
        size_t total_number = 1;
        for (size_t i = shape.size(); i-- > 0;)
        {
            total_number *= shape[i];
        }
        std::vector<int> data(total_number);
        std::random_device rd;                          // 随机数生成器种子
        std::mt19937 gen(rd());                         // 基于mt19937算法的随机数生成器
        std::uniform_int_distribution<int> dis(0, 100); // 定义一个从0到100的均匀分布

        for (int &val : data)
        {
            val = dis(gen); // 生成一个随机数并赋值
        }
        return Tensor<int>(data, shape);
    }

    // 创建所有元素为0的Tensor
    template <typename T>
    Tensor<T> Tensor<T>::zeros(const std::vector<size_t> &shape)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Shape cannot be empty.(when using zeros to create tensor)");
        }
        for (size_t i = 0; i < shape.size(); i++)
        {
            if (shape[i] == 0 || shape[i] > 1000)
            {
                throw std::invalid_argument(
                    "Shape dimensions must be greater than zero and should not be too large.(when using zeros to create tensor)");
            }
        }
        size_t total_number = 1;
        for (size_t i = shape.size(); i-- > 0;)
        {
            total_number *= shape[i];
        }
        return Tensor<T>(std::vector<T>(total_number, 0), shape);
    }

    // 创建所有元素为1的Tensor
    template <typename T>
    Tensor<T> Tensor<T>::ones(const std::vector<size_t> &shape)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Shape cannot be empty.(when using ones to create tensor)");
        }
        for (size_t i = 0; i < shape.size(); i++)
        {
            if (shape[i] == 0 || shape[i] > 1000)
            {
                throw std::invalid_argument(
                    "Shape dimensions must be greater than zero and should not be too large.(when using ones to create tensor)");
            }
        }
        size_t total_number = 1;
        for (size_t i = shape.size(); i-- > 0;)
        {
            total_number *= shape[i];
        }
        return Tensor<T>(std::vector<T>(total_number, 1), shape);
    }

    // 创建所有元素为指定值的Tensor
    template <typename T>
    Tensor<T> Tensor<T>::full(const std::vector<size_t> &shape, T value)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("Shape cannot be empty.(when using full to create tensor)");
        }

        for (size_t i = 0; i < shape.size(); i++)
        {
            if (shape[i] == 0 || shape[i] > 1000)
            {
                throw std::invalid_argument(
                    "Shape dimensions must be greater than zero and should not be too large.(when using full to create tensor)");
            }
        }
        size_t total_number = 1;
        for (size_t i = shape.size(); i-- > 0;)
        {
            total_number *= shape[i];
        }
        return Tensor<T>(std::vector<T>(total_number, value), shape);
    }

    // 创建单位矩阵，发现需要限制这些张量的形状大小（不清楚怎么处理），尝试输入负数会导致尺寸爆炸。
    template <typename T>
    Tensor<T> Tensor<T>::eye(size_t size)
    {
        if (size == 0 || size > 1000)
        {
            throw std::invalid_argument(
                "Shape dimensions must be greater than zero and should not be too large.(when using eye to create tensor)");
        }
        Tensor<T> mytensor = Tensor<T>::zeros({size, size});
        for (size_t i = 0; i < size; i++)
        {
            mytensor({i, i}) = 1;
        }
        return mytensor;
    }

    // 切片
    template <typename T>
    Tensor<T> Tensor<T>::slice(const std::vector<Slice> &slices) const
    {
        if (slices.size() > shape_.size())
        {
            throw std::invalid_argument("Number of slices cannot exceed tensor dimensions.");
        }

        // 新的形状和切片
        std::vector<size_t> new_shape;
        std::vector<Slice> new_slices;

        for (size_t i = 0; i < slices.size(); ++i)
        {
            // Slice对象的情况
            Slice s = slices[i];
            if (s.start >= s.end || s.end > shape_[i])
            {
                throw std::out_of_range("Invalid slice range for dimension " + std::to_string(i));
            }
            new_shape.push_back(s.end - s.start);
            new_slices.push_back(s);
        }
        // 对于未指定的维度，保持原始形状和切片
        for (size_t i = slices.size(); i < shape_.size(); ++i)
        {
            new_shape.push_back(shape_[i]);
            new_slices.push_back(Slice(0, shape_[i])); // 整个维度的切片
        }

        // 创建一个新的Tensor对象，共享相同的数据但具有不同的形状和切片
        return Tensor<T>(this->data_, new_shape, new_slices, strides_);
    }

    //     template <typename T>
    //     Tensor<T> cat(const std::vector<Tensor<T>> &tensors, size_t dim)
    //     {
    //         if (tensors.empty())
    //         {
    //             throw std::invalid_argument("Tensor list is empty.");
    //         }

    //         // 检查维度是否有效
    //         for (const auto &tensor : tensors)
    //         {
    //             if (dim >= tensor.get_shape().size())
    //             {
    //                 throw std::invalid_argument("Invalid dimension for concatenation.");
    //             }
    //         }

    //         // 检查连接的张量维度是否一致
    //         for (size_t i = 1; i < tensors.size(); ++i)
    //         {
    //             if (tensors[i].get_shape()[dim] != tensors[0].get_shape()[dim])
    //             {
    //                 throw std::invalid_argument("Invalid tensor shapes for concatenation.");
    //             }
    //         }

    //         // 计算新张量的形状
    //         std::vector<size_t> new_shape = tensors[0].get_shape();
    //         size_t total_dim_size = 0;
    //         for (const auto &tensor : tensors)
    //         {
    //             total_dim_size += tensor.get_shape()[dim];
    //         }
    //         new_shape[dim] = total_dim_size;

    //         // 创建新张量
    //         Tensor<T> result = Tensor<T>::zeros(new_shape);

    //         // 进行连接操作
    //         std::vector<size_t> indexes(new_shape.size(), 0);
    //         size_t start_index = 0;
    //         for (const auto &tensor : tensors)
    //         {
    //             recursiveCat(tensor, result, dim, indexes, 0, start_index);
    //             start_index += tensor.get_shape()[dim];
    //         }

    //         return result;
    //     }

    // template <typename T>
    // void recursiveCat(const Tensor<T> &input, Tensor<T> &output, size_t dim, std::vector<size_t> &indexes, size_t current_dim, size_t start_index)
    // {
    //     if (current_dim == dim)
    //     {
    //         for (size_t i = 0; i < input.get_shape()[dim]; ++i)
    //         {
    //             indexes[dim] = start_index + i;
    //             recursiveCat(input, output, dim, indexes, current_dim + 1, start_index);
    //         }
    //     }
    //     else if (current_dim < output.get_shape().size())
    //     {
    //         for (size_t i = 0; i < input.get_shape()[current_dim]; ++i)
    //         {
    //             indexes[current_dim] = i;
    //             recursiveCat(input, output, dim, indexes, current_dim + 1, start_index);
    //         }
    //     }
    //     else
    //     {
    //         auto input_index = indexes;
    //         input_index[dim] -= start_index; // 确保 output_index[dim] 不超过 input 的 dim 维度大小
    //         output(indexes) = input(input_index);
    //     }
    // }
    // template <typename T>
    // Tensor<T> tile(const Tensor<T> &tensor, const std::vector<size_t> &dims)
    // {
    //     if (dims.size() != tensor.get_shape().size())
    //     {
    //         throw std::invalid_argument("Dimensions for tiling do not match the tensor shape.");
    //     }

    //     Tensor<int> result = tensor;
    //     size_t count = 0;
    //     Tensor<int> my_tensor = tensor;
    //     for (size_t dim : dims)
    //     {
    //         for (size_t i = 1; i < dim; i++)
    //         {
    //             result = ts::cat<int>({result, my_tensor}, count);
    //         }
    //         count++;
    //         my_tensor = result;
    //     }
    //     return result;
    // }

    template <typename T>
    Tensor<T> Tensor<T>::transpose(size_t dim1, size_t dim2)
    {
        // 获取张量的形状
        std::vector<size_t> shape = get_shape();

        // 确保维度有效
        if (dim1 >= shape.size() || dim2 >= shape.size())
        {
            throw std::invalid_argument("Invalid dimensions for transpose.");
        }

        // 交换dim1和dim2维度的大小
        std::swap(shape[dim1], shape[dim2]);

        // 创建新的张量以进行转置
        Tensor<T> transposed_tensor = Tensor<T>::zeros(shape);

        // 执行转置操作
        std::vector<size_t> indexes(shape.size(), 0);
        for (size_t i = 0; i < get_shape()[0]; ++i)
        {
            indexes[dim1] = i;
            for (size_t j = 0; j < get_shape()[1]; ++j)
            {
                indexes[dim2] = j;
                transposed_tensor({j, i}) = (*this)(indexes);
            }
        }

        return transposed_tensor;
    }


    template <typename T>
    Tensor<T> Tensor<T>::view(const std::vector<size_t> &new_shape)
    {
        size_t total_size = 1;
        for (size_t dim : new_shape)
        {
            total_size *= dim;
        }
        if (total_size != data_->size())
        {
            throw std::invalid_argument("Total size of new shape must be equal to the original size.");
        }
        strides_.resize(new_shape.size());
        size_t stride = 1;
        for (int i = new_shape.size() - 1; i >= 0; i--)
        {
            strides_[i] = stride;
            stride *= new_shape[i];
        }
        shape_ = new_shape;
        return *this;
    }

    // 在Tensor类外部实现静态函数
    template <typename T>
    Tensor<T> Tensor<T>::view(const Tensor<T> &tensor, const std::vector<size_t> &new_shape)
    {

        Tensor<T> new_tensor(tensor.data_, tensor.shape_, tensor.slices_, tensor.strides_, 1);

        return new_tensor.view(new_shape);
    }
}

template class ts::Tensor<int>;
template class ts::Tensor<double>;
