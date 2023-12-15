#include <iostream>
#include "Tensor.h"
int main()
{
    //1.1
    Tensor<int> mytensor1({1,2,3,4,5,6,7,8},{2,4});
    mytensor1.print();
    //1.2
    Tensor<int> mytensor2=Tensor<int>::rand({1,2,3,4,5});
    Tensor<double> mytensor3=Tensor<double>::rand({1,2,3,4,5});
    mytensor2.print();
    mytensor3.print();
    //1.3
    Tensor<int> mytensor4=Tensor<int>::zeros({1,2,3});
    Tensor<int> mytensor5=Tensor<int>::ones({1,2,3});
    Tensor<int> mytensor6=Tensor<int>::full({1,2,3},9);
    Tensor<int> mytensor7=Tensor<int>::eye(0);

    mytensor4.print();
    mytensor5.print();
    mytensor6.print();
    mytensor7.print();


}