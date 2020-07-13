#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rl_msgs/SegmentationScene.h>

void fromRLMsg( rl_msgs::SegmentationObject &so_in, 
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud );


void fromRLMsg( rl_msgs::SegmentationFace &fc_in, 
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud );

void fromRLMsg( rl_msgs::SegmentationScene &scene,
				 pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud );