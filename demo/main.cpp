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
    clock_t t = clock();
    char * _time = new char[11];
    sprintf(_time, "%010d", t);
    std::cout << _time << "  " << t << "  " << CLOCKS_PER_SEC << std::endl;
}