#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <torch/torch.h>

int main(int argc, char const *argv[])
{
    cv::VideoCapture cap;
    if (!cap.open(0)) {
        return 0;
    }
    for (;;) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        cv::imshow("this is you, smile! :", frame);
        if (cv::waitKey(10) == 27) break; // stop capturing by pressing ESC
    }
    return 0;
}

