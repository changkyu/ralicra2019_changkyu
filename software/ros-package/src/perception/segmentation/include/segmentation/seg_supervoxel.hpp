#ifndef SEG_SUPERVOXEL__H__
#define SEG_SUPERVOXEL__H__

#include <pcl/point_types.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include "segmentation/seg_param.hpp"

void DoSupervoxelClustering(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud_ptr,
    SuperVoxelParam &param,
    std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr> *sv_clusters,
    std::multimap<uint32_t, uint32_t> *sv_adj=NULL,
    pcl::PointCloud<pcl::PointXYZL>::Ptr *sv_labeled_cloud=NULL,
    pcl::SupervoxelClustering<pcl::PointXYZRGB>::VoxelAdjacencyList *sv_adj_list=NULL
);

#endif