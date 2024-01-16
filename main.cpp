#include "Tensor.h"
//编译命令： g++ -o main main.cpp tensor.cpp -std=c++17 
//用了个variant,好像只有17支持.
using namespace ts;
int main()
{
    //1.1
    Tensor<int> mytensor1({1,2,3,4,5,6,7,8},{2,2,2});
    mytensor1.print();
    auto mytensor2=mytensor1.slice({1});
    mytensor2.print();
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