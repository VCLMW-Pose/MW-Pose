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
    int size[] = {2, 3, 4};

    Mat mat(3, size, CV_8U, Scalar(1));
    Mat test = mat(Range::all(), Range(0, 1));
    std::cout << test << std::endl;

    return 0;
}