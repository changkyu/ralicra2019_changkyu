#ifndef VIS__HPP__
#define VIS__HPP__

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

void addSupervoxelGraphToViewer(
    std::map <uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr > &supervoxel_clusters,
    std::multimap<uint32_t,uint32_t> &label_adjacency,
    std::map<std::pair<uint32_t,uint32_t>, float> &weight_adjacency,
    pcl::visualization::PCLVisualizer &viewer
);

void addSupervoxelIndexToViewer(
    std::map <uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr > supervoxel_clusters,
    pcl::visualization::PCLVisualizer &viewer
);

void addSupervoxelToViewer(  
  std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr> &supervoxel_clusters,
  pcl::visualization::PCLVisualizer &viewer );

void drawLabels(cv::Mat &markers, cv::Mat &dst, int idx_bg=-1);

#endif