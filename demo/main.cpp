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
    walabot test(10, 600, 10, -60, 60, 10, -30, 30, 10, true, 15);
    test.start();
    test.scan("D:\\Scan test\\", 100);
}