#include "Tensor.h"
#include "xtensor/xpad.hpp"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

using namespace ts;

int main() {
    //1.1
    Tensor<int> mytensor1({1,2,3},{3});
    //1.2 测试random
    Tensor<int> mytensor2 = Tensor<int>::rand({1, 2, 3, 4, 5});
    Tensor<double> mytensor3 = Tensor<double>::rand({1, 2, 3, 4, 5});
    std::cout << "----random test 1----" << std::endl;
    mytensor2.print();
    std::cout << "----random test 2----" << std::endl;
    mytensor3.print();
    //1.3 测试 zeros,ones,full,eye等
    Tensor<int> mytensor4 = Tensor<int>::zeros({1, 2, 3});
    Tensor<int> mytensor5 = Tensor<int>::ones({1, 2, 3});
    Tensor<int> mytensor6 = Tensor<int>::full({1, 2, 3}, 9);
    Tensor<int> mytensor7 = Tensor<int>::eye(2);
    std::cout << "----zeros test----" << std::endl;
    mytensor4.print();

    std::cout << "----zeros answer----" << std::endl;
    auto zero = xt::zeros<double>({1,2,3});
    std::cout << zero << std::endl;

    std::cout << "----ones test----" << std::endl;
    mytensor5.print();

    std::cout << "----ones answer----" << std::endl;
    auto one_test = xt::ones<double>({1,2,3});
    std::cout << one_test << std::endl;

    std::cout << "----full test----" << std::endl;
    mytensor6.print();

    std::cout << "----fulls answer----" << std::endl;
    auto ful_test = xt::xarray<double>({1,2,3},9);
    std::cout << ful_test << std::endl;

    std::cout << "----eye test----" << std::endl;
    mytensor7.print();

    std::cout << "----eye answer----" << std::endl;
    auto eye_test = xt::eye<double>({2,2});
    std::cout << eye_test << std::endl;

    Tensor<int> cat_tensor1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3});
    Tensor<int> cat_tensor2({13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, {2, 2, 3});
    std::cout << "tensor1 : " << std::endl;
    cat_tensor1.print();
    std::cout << std::endl;

    std::cout << "tensor2 : " << std::endl;
    cat_tensor2.print();
    std::cout << std::endl;

    //indexing
    std::cout << "index element of tensor[1]" << std::endl;
    Tensor<int> index_test = cat_tensor1(1) ;
    index_test.print();
    std::cout << "data address:" << get_data_address(index_test) << std::endl;

    //slicing
    std::cout << "slice test of tensor[1,{0,1}]" << std::endl;
    Tensor<int> index_test2 = cat_tensor1(1,{0,1});
    index_test2.print();
    std::cout << "data address :" << get_data_address(index_test2) << std::endl;
    std::cout << std::endl;

    // cat
    std::vector<Tensor<int>> tensors;
    tensors.push_back(cat_tensor1);
    tensors.push_back(cat_tensor2);
    Tensor<int> cat_tensor3 = cat(tensors, 0);
    std::cout << "cat mytensor1 and mytensor2 : " << std::endl;
    cat_tensor3.print();
    std::cout << std::endl;
    std::cout << "Correct answer" << std::endl;
    xt::xarray<int> a({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    a.reshape({2, 2, 3});
    xt::xarray<int> b{13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    b.reshape({2, 2, 3});
    auto ans = xt::concatenate(xt::xtuple(a, b), 0);
    std::cout << ans << std::endl;
    std::cout << std::endl;

    // tile
    Tensor<int> tile_tensor2({1, 2, 3, 4}, {2, 2});
    std::cout << "tile tensor" << std::endl;
    tile_tensor2.print();
    Tensor<int> tile_tensor3 = tile(tile_tensor2, {2, 2});
    std::cout << "tile  (A simple matrix ) in size 2*2 : " << std::endl;
    tile_tensor3.print();
    std::cout << "Correct answer" << std::endl;
    xt::xarray<int> tile_tes{1, 2, 3, 4};
    tile_tes.reshape({2, 2});
    auto tile_arr = xt::tile(tile_tes, {2, 2});
    std::cout << tile_arr << std::endl;
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

    std::cout << "correct answer" << std::endl;
    xt::xarray<int> trans_test = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    trans_test.reshape({3, 5});
    auto trans_arr = xt::transpose(trans_test);
    std::cout << trans_arr << std::endl;
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

    std::cout << "correct answer" << std::endl;
    auto permute_arr = xt::transpose(trans_test, {1, 0});
    std::cout << permute_arr << std::endl;

    Tensor<int> permute_tensor3({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3});
    std::cout << "permute (original shape high dim) : " << std::endl;
    permute_tensor3.print();
    std::cout << std::endl;

    Tensor<int> permute_tensor4 = permute_tensor3.permute({2, 1, 0});
    std::cout << "permute(high dim) : " << std::endl;
    permute_tensor4.print();
    std::cout << std::endl;

    xt::xarray<int> permute_test = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    permute_test.reshape({2, 2, 3});
    std::cout << "correct answer" << std::endl;
    auto permute_arr2 = xt::transpose(permute_test, {2, 1, 0});
    std::cout << permute_arr2 << std::endl;

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

    // Math operations test
    Tensor<int> t1({1, 2, 3, 4, 5, 6}, {3, 2});
    Tensor<int> t2({1, 2, 3}, {1, 3});
    Tensor<int> t3({2, 2, 4, 5, 5, 7}, {3, 2});
    xt::xarray<int> x1 = {1,2,3,4,5,6};
    x1.reshape({3, 2});
    xt::xarray<int> x2 = {1,2,3};
    x2.reshape({1,3});
    xt::xarray<int> x3 = {2,2,4,5,5,7};
    x3.reshape({3, 2});

    std::cout << "Math operations part: Each shows two operations and one test answer" << std::endl;
    std::cout << "Tensor No.1" << std::endl;
    t1.print();
    std::cout << "Tensor No.2" << std::endl;
    t3.print();

    std::cout << "----add----" << std::endl;
    Tensor<int> add1 = t1.add(t3);
    add1.print();
    Tensor<int> add2 = t1 + t3;
    add2.print();

    std::cout << "correct answer" << std::endl;
    std::cout << x1 + x3 << std::endl;

    std::cout << "----sub----" << std::endl;
    Tensor<int> sub1 = t1.sub(t3);
    sub1.print();
    Tensor<int> sub2 = t1 - t3;
    sub2.print();

    std::cout << "correct answer" << std::endl;
    std::cout << x1 - x3 << std::endl;

    std::cout << "----mul----" << std::endl;
    Tensor<int> mul1 = t1.mul(t3);
    mul1.print();
    Tensor<int> mul2 = t1 * t3;
    mul2.print();
    std::cout << "correct answer" << std::endl;
    std::cout << x1 * x3 << std::endl;

    std::cout << "----div----" << std::endl;
    Tensor<double> div1 = t1.div(t3);
    div1.print();
    Tensor<double> div2 = t1 / t3;
    div2.print();

    std::cout << "correct answer" << std::endl;
    xt::xarray<double> div_arr = xt::cast<double>(x1) / xt::cast<double>(x3);;
    std::cout << div_arr << std::endl;

    std::cout << "----log----" << std::endl;
    Tensor<double> log1 = log(t1);
    log1.print();

    std::cout << "correct answer" << std::endl;
    std::cout << xt::log(x1) << std::endl;

    std::cout << "----sum----" << std::endl;
    Tensor<double> sum1 = sum(t1, 0);
    sum1.print();
    Tensor<double> sum2 = t1.sum(0);
    sum2.print();

    std::cout << "correct answer" << std::endl;
    std::cout << xt::sum(x1,0) << std::endl;

    std::cout << "----mean----" << std::endl;
    Tensor<double> mean1 = mean(t1, 0);
    mean1.print();
    Tensor<double> mean2 = t1.mean(0);
    mean2.print();

    std::cout << "correct answer" << std::endl;
    std::cout << xt::mean(x1,0) << std::endl;

    std::cout << "----min----" << std::endl;
    Tensor<double> min1 = min(t1, 0);
    min1.print();
    Tensor<double> min2 = t1.min(0);
    min2.print();

    std::cout << "correct answer" << std::endl;
    std::cout << xt::amin(x1,0) << std::endl;

    std::cout << "----max----" << std::endl;
    Tensor<double> max1 = max(t1, 0);
    max1.print();
    Tensor<double> max2 = t1.max(0);
    max2.print();

    std::cout << "correct answer" << std::endl;
    std::cout << xt::amax(x1,0) << std::endl;

    std::cout << "----eq----" << std::endl;
    Tensor<int> eq1 = eq(t1, t3);
    eq1.print();

    std::cout << "correct answer" << std::endl;
    std::cout << xt::equal(x1,x3) << std::endl;

    std::cout << "----ne----" << std::endl;
    Tensor<int> ne1 = ne(t1, t3);
    ne1.print();

    std::cout << "correct answer" << std::endl;
    std::cout << xt::not_equal(x1,x3) << std::endl;

    std::cout << "----lt----" << std::endl;
    Tensor<int> lt1 = t1.lt(t3);
    lt1.print();

    std::cout << "correct answer" << std::endl;
    std::cout << xt::less(x1,x3) << std::endl;

    std::cout << "----le----" << std::endl;
    Tensor<int> le1 = t1.le(t3);
    le1.print();

    std::cout << "correct answer" << std::endl;
    std::cout << xt::less_equal(x1,x3) << std::endl;

    std::cout << "----gt----" << std::endl;
    Tensor<int> gt1 = t1.gt(t3);
    gt1.print();

    std::cout << "correct answer" << std::endl;
    std::cout << xt::greater(x1,x3) << std::endl;

    std::cout << "----ge----" << std::endl;
    Tensor<int> ge = t1.ge(t3);
    ge.print();

    std::cout << "correct answer" << std::endl;
    std::cout << xt::greater_equal(x1,x3) << std::endl;

    std::cout << "----broadcast----" << std::endl;
    std::vector<size_t> in = {3, 3};
    Tensor<int> output = broadcast(t2, in);
    output.print();

    std::cout << "correct answer" << std::endl;
    std::cout << xt::broadcast(x2,in) << std::endl;

    std::cout << "Serialization: save" << std::endl;
    Tensor<int> myTensor = Tensor<int>::rand({2, 4});
    myTensor.print();
    saveTensorToFile(myTensor, "my_tensor.bin");

    // 加载张量
    Tensor<int> loadedTensor = loadTensorFromFile<int>("my_tensor.bin");
    loadedTensor.print();

    std::cout << "----einsum----" << std::endl;
    try {

        {
            //2.transpose
            std::cout << "-------------Transpose---------------" << std::endl;
            auto A = Tensor<int>::rand({3, 6});
            std::vector<Tensor<int>> tensors = {A};
            std::string einsum_str = "ij->ji";
            Tensor<int> result = execute_einsum(tensors, einsum_str);
            A.print();
            result.print();
        }
        {
            //3.Permute
            std::cout << "-------------Permute---------------" << std::endl;
            auto A = Tensor<int>::rand({5, 6, 7, 8});
            std::vector<Tensor<int>> tensors = {A};
            std::string einsum_str = "...ij->...ji";
            Tensor<int> result = execute_einsum(tensors, einsum_str);
            A.print();
            result.print();
        }
        {
            //4.Reduce sum
            std::cout << "-------------Reduce sum---------------" << std::endl;
            auto A = Tensor<int>::rand({5, 6});
            std::vector<Tensor<int>> tensors = {A};
            std::string einsum_str = "ij->";
            Tensor<int> result = execute_einsum(tensors, einsum_str);
            A.print();
            result.print();
        }
        {
            //5.Sum along dimension
            std::cout << "-------------Sum along dimension---------------" << std::endl;
            auto A = Tensor<int>::rand({5, 6});
            std::vector<Tensor<int>> tensors = {A};
            std::string einsum_str = "ij->i";
            Tensor<int> result = execute_einsum(tensors, einsum_str);
            A.print();
            result.print();
        }
        {
            //6.Matrix and vector mul
            std::cout << "-------------Matrix and vector mul---------------" << std::endl;
            auto A = Tensor<int>::rand({5, 6});
            auto B = Tensor<int>::rand({6});
            std::vector<Tensor<int>> tensors = {A, B};
            std::string einsum_str = "ik,k->i";
            Tensor<int> result = execute_einsum(tensors, einsum_str);
            A.print();
            B.print();
            result.print();
        }
        {
            //7.Matrix mul
            std::cout << "-------------Matrix mull---------------" << std::endl;
            auto A = Tensor<int>::rand({5, 6});
            auto B = Tensor<int>::rand({6, 7});
            std::vector<Tensor<int>> tensors = {A, B};
            std::string einsum_str = "ik,kj->ij";
            Tensor<int> result = execute_einsum(tensors, einsum_str);
            A.print();
            B.print();
            result.print();
        }
        {
            //8. Dot product
            std::cout << "------------- Dot product---------------" << std::endl;
            auto A = Tensor<int>::rand({6});
            auto B = Tensor<int>::rand({6});
            std::vector<Tensor<int>> tensors = {A, B};
            std::string einsum_str = "i,i->";
            Tensor<int> result = execute_einsum(tensors, einsum_str);
            A.print();
            B.print();
            result.print();
        }
        {
            //9. Pointwise mul and reduce sum
            std::cout << "------------- Pointwise mul and reduce sum---------------" << std::endl;
            auto A = Tensor<int>::rand({5, 6});
            auto B = Tensor<int>::rand({5, 6});
            std::vector<Tensor<int>> tensors = {A, B};
            std::string einsum_str = "ij,ij->";
            Tensor<int> result = execute_einsum(tensors, einsum_str);
            A.print();
            B.print();
            result.print();
        }
        {
            //10.  Outer product
            std::cout << "-------------  Outer product---------------" << std::endl;
            auto A = Tensor<int>::rand({5});
            auto B = Tensor<int>::rand({6});
            std::vector<Tensor<int>> tensors = {A, B};
            std::string einsum_str = "i,j->ij";
            Tensor<int> result = execute_einsum(tensors, einsum_str);
            A.print();
            B.print();
            result.print();
        }
        {
            //11.  Batch matrix mul
            std::cout << "------------- Batch matrix mul---------------" << std::endl;
            auto A = Tensor<int>::rand({5, 6, 7});
            auto B = Tensor<int>::rand({5, 7, 8});
            std::vector<Tensor<int>> tensors = {A, B};
            std::string einsum_str = "ijk,ikl->ijl";
            Tensor<int> result = execute_einsum(tensors, einsum_str);
            A.print();
            B.print();
            result.print();
        }
        {
            //12.  Tensor contraction
            std::cout << "------------- Batch matrix mul---------------" << std::endl;
            auto A = Tensor<int>::rand({5, 6, 7, 8});
            auto B = Tensor<int>::rand({4, 5, 6, 7, 7});
            std::vector<Tensor<int>> tensors = {A, B};
            std::string einsum_str = "pqrs,tuqvr->pstuv";
            Tensor<int> result = execute_einsum(tensors, einsum_str);
            A.print();
            B.print();
            result.print();
        }
        {
            //13.  Bilinear transformation
            std::cout << "-------------  Bilinear transformation---------------" << std::endl;
            auto A = Tensor<int>::rand({6, 7});
            auto B = Tensor<int>::rand({5, 7, 8});
            std::vector<Tensor<int>> tensors = {A, B};
            std::string einsum_str = "ik,jkl->ij";
            Tensor<int> result = execute_einsum(tensors, einsum_str);
            A.print();
            B.print();
            result.print();
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }


}
