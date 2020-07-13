#include "segmentation/seg_supervoxel.hpp"

using namespace std;
using namespace pcl;

void DoSupervoxelClustering(
    PointCloud<PointXYZRGB>::Ptr &input_cloud_ptr,
    SuperVoxelParam &param,
    map<uint32_t, Supervoxel<PointXYZRGB>::Ptr> *sv_clusters,
    multimap<uint32_t, uint32_t> *sv_adj,
    PointCloud<PointXYZL>::Ptr *sv_labeled_cloud,
    SupervoxelClustering<PointXYZRGB>::VoxelAdjacencyList *sv_adj_list
    )
{
    SupervoxelClustering<PointXYZRGB> 
        super(param.voxel_resolution, param.seed_resolution);
    super.setUseSingleCameraTransform (param.use_single_cam_transform);
    super.setInputCloud (input_cloud_ptr);
//    if (has_normals) super.setNormalCloud (input_normals_ptr);
    super.setColorImportance (param.color_importance);
    super.setSpatialImportance (param.spatial_importance);
    super.setNormalImportance (param.normal_importance);
    super.extract (*sv_clusters);

    if (param.use_supervoxel_refinement)
    {
        super.refineSupervoxels (2, *sv_clusters);
    }

    if(sv_adj)
        super.getSupervoxelAdjacency (*sv_adj);

    if(sv_labeled_cloud)
        *sv_labeled_cloud = super.getLabeledCloud ();

    if(sv_adj_list)
        super.getSupervoxelAdjacencyList(*sv_adj_list);
}
