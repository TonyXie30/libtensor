#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include "Tensor.h"
using namespace ts;
// void parse_einsum_str(
//     const std::string& einsum_str,
//     std::map<char, std::map<int, int>>& index_dim_map, // 输入张量维度映射
//     std::map<char, int>& output_index_dim_map,         // 输出张量维度映射
//     std::vector<char>& indices_order                   // 索引字符的顺序
// ) {
//     index_dim_map.clear();
//     output_index_dim_map.clear();
//     indices_order.clear();

//     auto arrow_pos = einsum_str.find("->");
//     auto input_str = einsum_str.substr(0, arrow_pos);
//     auto output_str = einsum_str.substr(arrow_pos + 2);

//     int tensor_index = 0;
//     int dim_index = 0;
//     for (char ch : input_str) {
//         if (ch == ',') {
//             tensor_index++;
//             dim_index = 0;
//             continue;
//         }
//         if (ch != ' ') {
//             index_dim_map[ch][tensor_index] = dim_index++;
//             if (std::find(indices_order.begin(), indices_order.end(), ch) == indices_order.end()) {
//                 indices_order.push_back(ch);
//             }
//         }
//     }

//     // 解析输出字符串，记录输出张量中每个索引的维度位置
//     dim_index = 0;
//     for (char ch : output_str) {
//         if (ch != ' ') {
//             output_index_dim_map[ch] = dim_index++;
//         }
//     }
// }
// void parse_einsum_str(
//     const std::string& einsum_str,
//     std::map<char, std::map<int, int>>& index_dim_map, // 输入张量维度映射
//     std::map<char, int>& output_index_dim_map,         // 输出张量维度映射
//     std::vector<char>& indices_order                   // 索引字符的顺序
// ) {
//     index_dim_map.clear();
//     output_index_dim_map.clear();
//     indices_order.clear();

//     auto arrow_pos = einsum_str.find("->");
//     auto input_str = einsum_str.substr(0, arrow_pos);
//     // 确保 "->" 后面有字符
//     auto output_str = (arrow_pos != std::string::npos && arrow_pos + 2 < einsum_str.length()) 
//                       ? einsum_str.substr(arrow_pos + 2) 
//                       : "";

//     int tensor_index = 0;
//     int dim_index = 0;
//     for (char ch : input_str) {
//         if (ch == ',') {
//             tensor_index++;
//             dim_index = 0;
//             continue;
//         }
//         if (ch != ' ') {
//             index_dim_map[ch][tensor_index] = dim_index++;
//             if (std::find(indices_order.begin(), indices_order.end(), ch) == indices_order.end()) {
//                 indices_order.push_back(ch);
//             }
//         }
//     }

//     // 解析输出字符串，记录输出张量中每个索引的维度位置
//     if (!output_str.empty()) {
//         dim_index = 0;
//         for (char ch : output_str) {
//             if (ch != ' ') {
//                 output_index_dim_map[ch] = dim_index++;
//             }
//         }
//     }
// }
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
//创建输出张量
// template<typename T>
// Tensor<T> create_output_tensor(
//     const std::vector<Tensor<T>>& tensors,
//     const std::map<char, std::map<int, int>>& index_dim_map,
//     const std::map<char, int>& output_index_dim_map
// ) {
//     std::vector<size_t> shape;
//     for (const auto& [index, output_dim] : output_index_dim_map) {
//         if (index_dim_map.find(index) == index_dim_map.end()) {
//             throw std::runtime_error("Index not found in index_dim_map.");
//         }

//         // 使用第一个出现的张量和维度
//         const auto& tensor_dims = index_dim_map.at(index); // 获取与索引相关联的张量和维度的映射
//         auto tensor_index = tensor_dims.begin()->first;    // 获取第一个出现的张量索引
//         auto dim_index = tensor_dims.begin()->second;      // 获取对应的维度索引


//         if (tensor_index >= tensors.size()) {
//             throw std::runtime_error("Tensor index out of range.");
//         }

//         size_t dim_size = tensors[tensor_index].get_shape()[dim_index];
//         shape.push_back(dim_size);
//     }
//     return Tensor<T>::zeros(shape); // 使用形状创建新的 Tensor
// }
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
            sum *= tensors[i](input_indices[i]); // 累加来自每个张量的值
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

int main() {
    // // std::string einsum_str = "ijkl,ijmn->klmn";
    // std::string einsum_str = "ik,k->i";
    // // std::string einsum_str = "ij,jk,kl->il";
    // std::map<char, std::map<int, int>> index_dim_map; //各个字母对应各个张量的维度位置
    // std::map<char, int> output_index_dim_map; //各个字母对应输出张量的维度位置
    // std::vector<char> indices_order; //所有字母的集合，还有字母顺序
    // parse_einsum_str(einsum_str, index_dim_map, output_index_dim_map, indices_order);
    // // std::cout<<index_dim_map.at('j').begin()->second<<std::endl;
    // // std::cout<<output_index_dim_map.at('i')<<std::endl;
    // auto A= Tensor<int>::rand({2,3});
    // auto B= Tensor<int>::rand({3});
    // auto C= Tensor<int>::rand({4,2});
    // std::vector<Tensor<int>> tensors = {A,B};
    // if (!check_dimension_consistency(tensors, index_dim_map)) {//检查对应维度长度是否相同
    //     std::cerr << "Dimension mismatch among tensors." << std::endl;
    //     return 1;
    // }
    // std::vector<int> dim_lengths;//各个字母对应维度的长度
    // dim_lengths=generate_dim_lengths(tensors,indices_order,index_dim_map);
    // for (auto i:generate_dim_lengths(tensors,indices_order,index_dim_map))
    // {
    //     std::cout<<i<<" "; 
    // }
    // std::cout<<std::endl;
    // auto result =create_output_tensor(tensors,indices_order,dim_lengths,output_index_dim_map);//输出张量
    // for (auto i : result.get_shape())
    // {
    //     std::cout<<i<<std::endl;
    // }
    // std::vector<int> current_indices;//当前各个字母的对应的迭代中的索引
    // for (size_t i = 0; i < indices_order.size(); i++)
    // {
    //     current_indices.push_back(0);
    // }
    // recursive_einsum(tensors,result,indices_order,current_indices,output_index_dim_map,index_dim_map,dim_lengths);
    // A.print();
    // B.print();
    // C.print();
    // result.print();
    try {
        // // 示例用法
        // auto A = Tensor<int>::rand({2,3,4});
        // auto B = Tensor<int>::rand({3});
        // std::vector<Tensor<int>> tensors = {A};
        // std::string einsum_str = "...jk->...kj";
        // std::string einsum_str1 = "i,i->";

        // Tensor<int> result = execute_einsum(tensors, einsum_str);
        // // Tensor<int> result1 = execute_einsum(tensors, einsum_str1);

        // // 打印结果
        // A.print();
        // result.print();
        // // result1.print();
        {
        //2.transpose
        std::cout<<"-------------Transpose---------------"<<std::endl;
        auto A=Tensor<int>::rand({3,6});
        std::vector<Tensor<int>> tensors = {A};
        std::string einsum_str = "ij->ji";
        Tensor<int> result = execute_einsum(tensors, einsum_str);
        A.print();
        result.print();
        }
        {
        //3.Permute
        std::cout<<"-------------Permute---------------"<<std::endl;
        auto A=Tensor<int>::rand({5,6,7,8});
        std::vector<Tensor<int>> tensors = {A};
        std::string einsum_str = "...ij->...ji";
        Tensor<int> result = execute_einsum(tensors, einsum_str);
        A.print();
        result.print();
        }
        {
        //4.Reduce sum
        std::cout<<"-------------Reduce sum---------------"<<std::endl;
        auto A=Tensor<int>::rand({5,6});
        std::vector<Tensor<int>> tensors = {A};
        std::string einsum_str = "ij->";
        Tensor<int> result = execute_einsum(tensors, einsum_str);
        A.print();
        result.print();
        }
        {
        //5.Sum along dimension
        std::cout<<"-------------Sum along dimension---------------"<<std::endl;
        auto A=Tensor<int>::rand({5,6});
        std::vector<Tensor<int>> tensors = {A};
        std::string einsum_str = "ij->i";
        Tensor<int> result = execute_einsum(tensors, einsum_str);
        A.print();
        result.print();
        }
        {
        //6.Matrix and vector mul
        std::cout<<"-------------Matrix and vector mul---------------"<<std::endl;
        auto A=Tensor<int>::rand({5,6});
        auto B=Tensor<int>::rand({6});
        std::vector<Tensor<int>> tensors = {A,B};
        std::string einsum_str = "ik,k->i";
        Tensor<int> result = execute_einsum(tensors, einsum_str);
        A.print();
        B.print();
        result.print();
        }
        {
        //7.Matrix mul
        std::cout<<"-------------Matrix mull---------------"<<std::endl;
        auto A=Tensor<int>::rand({5,6});
        auto B=Tensor<int>::rand({6,7});
        std::vector<Tensor<int>> tensors = {A,B};
        std::string einsum_str = "ik,kj->ij";
        Tensor<int> result = execute_einsum(tensors, einsum_str);
        A.print();
        B.print();
        result.print();
        }
        {
        //8. Dot product
        std::cout<<"------------- Dot product---------------"<<std::endl;
        auto A=Tensor<int>::rand({6});
        auto B=Tensor<int>::rand({6});
        std::vector<Tensor<int>> tensors = {A,B};
        std::string einsum_str = "i,i->";
        Tensor<int> result = execute_einsum(tensors, einsum_str);
        A.print();
        B.print();
        result.print();
        }
        {
        //9. Pointwise mul and reduce sum
        std::cout<<"------------- Pointwise mul and reduce sum---------------"<<std::endl;
        auto A=Tensor<int>::rand({5,6});
        auto B=Tensor<int>::rand({5,6});
        std::vector<Tensor<int>> tensors = {A,B};
        std::string einsum_str = "ij,ij->";
        Tensor<int> result = execute_einsum(tensors, einsum_str);
        A.print();
        B.print();
        result.print();
        }
        {
        //10.  Outer product
        std::cout<<"-------------  Outer product---------------"<<std::endl;
        auto A=Tensor<int>::rand({5});
        auto B=Tensor<int>::rand({6});
        std::vector<Tensor<int>> tensors = {A,B};
        std::string einsum_str = "i,j->ij";
        Tensor<int> result = execute_einsum(tensors, einsum_str);
        A.print();
        B.print();
        result.print();
        }
        {
        //11.  Batch matrix mul
        std::cout<<"------------- Batch matrix mul---------------"<<std::endl;
        auto A=Tensor<int>::rand({5,6,7});
        auto B=Tensor<int>::rand({5,7,8});
        std::vector<Tensor<int>> tensors = {A,B};
        std::string einsum_str = "ijk,ikl->ijl";
        Tensor<int> result = execute_einsum(tensors, einsum_str);
        A.print();
        B.print();
        result.print();
        }
        {
        //12.  Tensor contraction
        std::cout<<"------------- Batch matrix mul---------------"<<std::endl;
        auto A=Tensor<int>::rand({5,6,7,8});
        auto B=Tensor<int>::rand({4,5,6,7,7});
        std::vector<Tensor<int>> tensors = {A,B};
        std::string einsum_str = "pqrs,tuqvr->pstuv";
        Tensor<int> result = execute_einsum(tensors, einsum_str);
        A.print();
        B.print();
        result.print();
        }  
        {
        //13.  Bilinear transformation
        std::cout<<"-------------  Bilinear transformation---------------"<<std::endl;
        auto A=Tensor<int>::rand({6,7});
        auto B=Tensor<int>::rand({5,7,8});
        std::vector<Tensor<int>> tensors = {A,B};
        std::string einsum_str = "ik,jkl->ij";
        Tensor<int> result = execute_einsum(tensors, einsum_str);
        A.print();
        B.print();
        result.print();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    

    return 0;
}