#ifndef GEN_GRAPH__H__
#define GEN_GRAPH__H__

#include <set>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

void ConnectEdges(pcl::PointCloud<pcl::PointXYZRGBL> &cloud_seg,
                  std::set<std::pair<uint32_t,uint32_t> > &edges );

#endif
