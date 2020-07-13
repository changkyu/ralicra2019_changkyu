#include <pcl/visualization/cloud_viewer.h>
#include "rl_msgs/SegmentationObject.h"

void AddSegmentationObjects(std::vector<rl_msgs::SegmentationObject> &segs,
                            pcl::visualization::PCLVisualizer &viewer, int viewport=0,
                            const Eigen::Matrix4f &tf=Eigen::Affine3f::Identity().matrix());
void AddSegmentationObjects2(std::vector<rl_msgs::SegmentationObject> &segs,
                            pcl::visualization::PCLVisualizer &viewer, int viewport=0,
                            const Eigen::Matrix4f &tf=Eigen::Affine3f::Identity().matrix());
void AddSegmentationObject( rl_msgs::SegmentationObject &seg,
                            pcl::visualization::PCLVisualizer &viewer, int idx, int viewport=0,
                            const Eigen::Matrix4f &tf=Eigen::Affine3f::Identity().matrix());
void AddSegmentationObject2( rl_msgs::SegmentationObject &segsv,
                            pcl::visualization::PCLVisualizer &viewer, int idx, int viewport=0,
                            const Eigen::Matrix4f &tf=Eigen::Affine3f::Identity().matrix());