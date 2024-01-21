#include "Tensor.h"

using namespace ts;
int main()
{
    //1.1
        // cat
    Tensor<int> cat_tensor1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3});
    Tensor<int> cat_tensor2({13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, {2, 2, 3});
    std::cout << "mytensor1 : " << std::endl;
    cat_tensor1.print();
    std::cout << std::endl;

    std::cout << "mytensor2 : " << std::endl;
    cat_tensor2.print();
    std::cout << std::endl;

    std::vector<Tensor<int>> tensors;
    tensors.push_back(cat_tensor1);
    tensors.push_back(cat_tensor2);
    Tensor<int> cat_tensor3 = cat(tensors, 0);
    std::cout << "cat mytensor1 and mytensor2 : " << std::endl;
    cat_tensor3.print();
    std::cout << std::endl;

    // tile
    Tensor<int> tile_tensor1 = tile(cat_tensor1, {2, 2});
    std::cout << "tile  in size 2*2 : " << std::endl;
    tile_tensor1.print();
    std::cout << std::endl;

    Tensor<int> tile_tensor2({1, 2, 3, 4}, {2, 2});
    Tensor<int> tile_tensor3 = tile(tile_tensor2, {2, 2});
    std::cout << "tile  (A simple matrix ) in size 2*2 : " << std::endl;
    tile_tensor3.print();
    std::cout << std::endl;

    // mutate
    Tensor<int> mutate_tensor1 = Tensor<int>::ones({2, 6});
    mutate_tensor1(1) = 5;
    std::cout << "mutating(sets the second element of tensor1 to 5) : " << std::endl;
    mutate_tensor1.print();
    auto mutate_addres1 = get_data_address(mutate_tensor1);
    std::cout << "Address of data: " << mutate_addres1 << std::endl;
    std::cout << std::endl;

    mutate_tensor1(1, {2, 4}) = {2, 3};
    std::cout << "mutating : " << std::endl;
    mutate_tensor1.print();
    auto mutate_addres2 = get_data_address(mutate_tensor1);
    std::cout << "Address of data: " << mutate_addres2 << std::endl;
    std::cout << std::endl;

    // transpose
    Tensor<int> transpose_tensor1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {3, 5});
    std::cout << "transpose_tensor1 (original shape) : " << std::endl;
    transpose_tensor1.print();
    std::cout << std::endl;

    Tensor<int> transpose_tensor2 = transpose_tensor1.transpose(0, 1);
    std::cout << "transpose_tensor1 (change dim 0 and 1) : " << std::endl;
    transpose_tensor2.print();
    std::cout << std::endl;

    // permute
    Tensor<int> permute_tensor1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {3, 5});
    std::cout << "permute (original shape) : " << std::endl;
    permute_tensor1.print();
    auto permute_addres1 = get_data_address(permute_tensor1);
    std::cout << "Address of data: " << permute_addres1 << std::endl;
    std::cout << std::endl;

    Tensor<int> permute_tensor2 = permute_tensor1.permute({1, 0});
    std::cout << "permute : " << std::endl;
    permute_tensor2.print();
    auto permute_addres2 = get_data_address(permute_tensor2);
    std::cout << "Address of data: " << permute_addres2 << std::endl;
    std::cout << std::endl;

    Tensor<int> permute_tensor3({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3});
    std::cout << "permute (original shape high dim) : " << std::endl;
    permute_tensor3.print();
    std::cout << std::endl;

    Tensor<int> permute_tensor4 = permute_tensor3.permute({2, 1, 0});
    std::cout << "permute(high dim) : " << std::endl;
    permute_tensor4.print();
    std::cout << std::endl;

    // view
    Tensor<int> view_tensor1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3});
    std::cout << "view (original shape) : " << std::endl;
    view_tensor1.print();
    auto view_addres1 = get_data_address(view_tensor1);
    std::cout << "Address of data: " << view_addres1 << std::endl;
    std::cout << std::endl;

    Tensor<int> view_tensor2 = view_tensor1.view({1, 12});
    std::cout << "view (change the shape) : " << std::endl;
    view_tensor2.print();
    auto view_addres2 = get_data_address(view_tensor2);
    std::cout << "Address of data: " << view_addres2 << std::endl;
    std::cout << std::endl;

    Tensor<int> view_tensor3 = view_tensor1.view({2, 2, 3});
    std::cout << "view (change the shape to high dim) : " << std::endl;
    view_tensor3.print();
    auto view_addres3 = get_data_address(view_tensor3);
    std::cout << "Address of data: " << view_addres3 << std::endl;
    std::cout << std::endl;

        Tensor<int> myTensor = Tensor<int>::rand({12});
    myTensor.print();
    saveTensorToFile(myTensor, "my_tensor.bin");

    // 加载张量
    Tensor<int> loadedTensor = loadTensorFromFile<int>("my_tensor.bin");
    loadedTensor.print();

    
//    //1.2
//    Tensor<int> mytensor2=Tensor<int>::rand({1,2,3,4,5});
//    Tensor<double> mytensor3=Tensor<double>::rand({1,2,3,4,5});
//    mytensor2.print();
//    //mytensor3.print();
//    //1.3
//    Tensor<int> mytensor4=Tensor<int>::zeros({1,2,3});
//    Tensor<int> mytensor5=Tensor<int>::ones({1,2,3});
//    Tensor<int> mytensor6=Tensor<int>::full({1,2,3},9);
//    //Tensor<int> mytensor7=Tensor<int>::eye(0);
//    mytensor4.print();
//    // mytensor5.print();
//    // mytensor6.print();
//    // mytensor7.print();
    // Example usage
    // Tensor<double> t1 = Tensor<double>::rand({3, 3});
    // t1.print();
    // Tensor<double> adder = Tensor<double>::rand({3,3});
    // adder.print();
    // Tensor<double> answer1 = t1.sub(adder);
    // answer1.print();
    // Tensor<double> answer2 = t1 -adder;
    // answer2.print();
    // Tensor<double> answer3 = t1.sub(3.0);
    // answer3.print();
}
