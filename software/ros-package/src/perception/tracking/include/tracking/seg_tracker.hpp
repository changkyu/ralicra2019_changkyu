#ifndef SEG_TRACKER__HPP__
#define SEG_TRACKER__HPP__

#include <vector>
#include <map>
#include <set>
#include <opencv2/core/core.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

void SegTrajectories(std::vector<cv::Mat> &images_seg, 
                     std::vector<std::vector<cv::Point2f> > &points_track,
                     std::vector<std::set<std::pair<size_t,size_t> > > &traj );

#endif