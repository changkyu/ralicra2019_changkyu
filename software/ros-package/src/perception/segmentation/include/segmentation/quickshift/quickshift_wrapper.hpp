#ifndef QUICKSHIFT_WRAPPER__HPP__
#define QUICKSHIFT_WRAPPER__HPP__

#include <opencv2/core/core.hpp>

void Segmentation_QuickShift(cv::Mat &image, cv::Mat &image_seg, 
                             float sigma=6, float tau=10);

#endif