#ifndef SEG_PREPROCESS__H__
#define SEG_PREPROCESS__H__

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/filter.h>
#include <shape_msgs/Plane.h>

void RemoveBackground(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_inout,
                      std::vector<shape_msgs::Plane> &planes_bg,
                      const std::vector<float> &camera_RT=std::vector<float>(), 
                      const int num_of_remove=1);
void RemoveInvalidPoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr output);
void RemoveNotInPlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_inout, 
                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rm    );

void RemoveBackground( pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_inout,
                       const std::vector<float> &workspace,
                       const std::vector<float> &camera_RT            );

#endif