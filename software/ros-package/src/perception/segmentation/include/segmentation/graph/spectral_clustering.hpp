#ifndef SPECTRAL_CLUSTERING__H__
#define SPECTRAL_CLUSTERING__H__

#include <vector>
#include <opencv2/core/core.hpp>

void SpectralClustering(cv::Mat &W, cv::Mat &D, std::vector<uint32_t> &labels_out, int num_of_objects=-1);
void SpectralClustering(cv::Mat &W, std::vector<uint32_t> &labels_out, int num_of_objects=-1);

#endif