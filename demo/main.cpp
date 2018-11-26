//
// Created by K Knight on 11/24/2018.
//

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "walabot.hpp"
using namespace cv;


int main()
{
    int size[] = {2, 12};
    walabot wlbt(0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0);
    int scalar[] = {1, 2, 3, 4 ,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ,16, 17, 18, 19, 20, 21, 22, 23, 24};

    Mat mat(2, 12, CV_32S, scalar);
    std::cout << mat << std::endl;
    Mat test = wlbt._sum_horizontal(mat, 2, 3, 4);
    std::cout << test << std::endl;
    std::cout << mat << std::endl;
    return 0;
}