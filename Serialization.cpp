#include "Tensor.h"
#include <fstream>

namespace ts{
    template<typename T>
    Tensor<T> save(Tensor<T> tensor, std::string filename) {
        // 序列化为二进制
        {
            std::ofstream os(filename, std::ios::binary);
            cereal::BinaryOutputArchive archive(os);
            archive(tensor);
        }

        return tensor; // 返回传入的 tensor（如果需要）
    }

    template<typename T>
    Tensor<T> load(std::string filename) {
        Tensor<T> tensor;

        // 从二进制反序列化
        {
            std::ifstream is(filename, std::ios::binary);
            cereal::BinaryInputArchive archive(is);
            archive(tensor);
        }

        return tensor;
    }
}

template
class ts::Tensor<int>;

template
class ts::Tensor<double>;

template
ts::Tensor<int> ts::save(ts::Tensor<int> tensor, std::string filename);

template
ts::Tensor<double> ts::save(ts::Tensor<double> tensor, std::string filename);

template
ts::Tensor<int> ts::load(std::string filename);

template
ts::Tensor<double> ts::load(std::string filename);
