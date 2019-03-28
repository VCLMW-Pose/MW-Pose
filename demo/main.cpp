//
// Created by K Knight on 11/24/2018.
//

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>----------------------------
#include <iostream>
#include "walabot.hpp"
using namespace cv;

int main()
{
    /*
    auto signal = "1";
    auto inter = fopen("F:\\capturedata\\inter.txt", "w");                                                                // Send signal
    fwrite(signal, sizeof(char), 1, inter);
    fclose(inter);*/

    walabot test(10, 600, 10, -60, 60, 10, -60, 60, 10, true, 15);
    test.start();
    test.union_scan("F:\\capture\\", 100);
}
