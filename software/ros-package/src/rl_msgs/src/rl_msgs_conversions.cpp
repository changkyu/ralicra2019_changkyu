#include <pcl_conversions/pcl_conversions.h>
#include <rl_msgs/rl_msgs_conversions.hpp>

using namespace std;
using namespace pcl;

void fromRLMsg( rl_msgs::SegmentationObject &so_in, 
                 PointCloud<PointXYZRGB>::Ptr cloud )
{
    fromROSMsg<PointXYZRGB>(so_in.cloud, *cloud);
}

void fromRLMsg( rl_msgs::SegmentationFace &fc_in, 
                 PointCloud<PointXYZRGB>::Ptr cloud )
{
    fromROSMsg<PointXYZRGB>(fc_in.cloud, *cloud);
}

void fromRLMsg( rl_msgs::SegmentationScene &scene,
				 PointCloud<PointXYZRGB>::Ptr cloud )
{
	for( size_t o=0; o<scene.objects.size(); o++ )
	{
		PointCloud<PointXYZRGB>::Ptr cloud_obj(new PointCloud<PointXYZRGB>);		
		fromRLMsg(scene.objects[o], cloud_obj);

		*cloud += *cloud_obj;
	}
}