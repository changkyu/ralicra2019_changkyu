#include <sstream>
#include <boost/make_shared.hpp>

//////////////////////////////////////////////////////////////////////////////
// ROS
//////////////////////////////////////////////////////////////////////////////
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>

//////////////////////////////////////////////////////////////////////////////
// PCL
//////////////////////////////////////////////////////////////////////////////
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/common/pca.h>

//////////////////////////////////////////////////////////////////////////////
// OpenCV
//////////////////////////////////////////////////////////////////////////////
#include <cv_bridge/cv_bridge.h>

//////////////////////////////////////////////////////////////////////////////
// OTHER
//////////////////////////////////////////////////////////////////////////////
#include "rl_msgs/seg_scene_srv.h"

#include "segmentation/quickshift/quickshift_wrapper.hpp"
#include "segmentation/graph/spectral_clustering.hpp"
#include "segmentation/seg_param.hpp"
#include "segmentation/seg_preprocess.hpp"
#include "segmentation/seg_supervoxel.hpp"
#include "segmentation/seg_lccp_2Dseg.hpp"
#include "segmentation/vis.hpp"

using namespace std;
using namespace pcl;
using namespace cv;

ros::NodeHandle* nh;

ros::Publisher pub_seg;
ros::Publisher pub_seg_wCOB;

string PUB_LCCP;
string SRV_LCCP;
string PUB_LCCP_2DSEG;
string SRV_LCCP_2DSEG;
string SUB_IMAGE;
string SUB_DEPTH;
string SUB_POINT;
string SUB_COB;
string SUB_CAMINFO;

vector<float> camera_K;
vector<float> camera_RT;

SegLCCP2DSeg* seglccp2dseg;

void ParseParam(ros::NodeHandle nh)
{
    nh.param<string>("/segmentation/lccp/pub_name",PUB_LCCP,"segmentation/lccp");
    nh.param<string>("/segmentation/lccp/srv_name",SRV_LCCP,"segmentation/lccp");

    nh.param<string>("/segmentation/lccp_seg/pub_name",PUB_LCCP_2DSEG,"segmentation/lccp_2Dseg");
    nh.param<string>("/segmentation/lccp_seg/srv_name",SRV_LCCP_2DSEG,"segmentation/lccp_2Dseg");

    nh.param<string>("/segmentation/topic_image",SUB_IMAGE, "/rgb/image");
    nh.param<string>("/segmentation/topic_depth",SUB_DEPTH, "/depth/image");
    nh.param<string>("/segmentation/topic_point",SUB_POINT, "/pointcloud");
    nh.param<string>("/segmentation/topic_segcob",SUB_COB,  "/segmentation/cob");
    nh.param<string>("/segmentation/topic_caminfo",SUB_CAMINFO, "/depth/camera_info");

    nh.getParam("/segmentation/camera_info/intrinsic", camera_K);
    nh.getParam("/segmentation/camera_info/extrinsic", camera_RT);
}

bool GetPointCloudFromCamera(PointCloud<PointXYZRGB>::Ptr cloud)
{        
    try
    {
        sensor_msgs::PointCloud2::ConstPtr msg_pc
         = ros::topic::waitForMessage<sensor_msgs::PointCloud2>(SUB_POINT, *nh);
        fromROSMsg<pcl::PointXYZRGB>(*msg_pc, *cloud);
    }
    catch(std::exception &e)
    {
        ROS_ERROR("Exception during waitForMessage PointCloud2: %s", e.what());
        return false;
    }
    return true;
}

bool GetImageFromCamera(sensor_msgs::Image &msg_img)
{
    try
    {
        sensor_msgs::Image::ConstPtr msg_ptr;
        msg_ptr = ros::topic::waitForMessage<sensor_msgs::Image>(SUB_IMAGE, *nh);
        msg_img = *msg_ptr;
    }
    catch(std::exception &e)
    {
        ROS_ERROR("Exception during waitForMessage Image: %s", e.what());
        return false;
    }
    return true;
}

bool Segmentation_lccp_2Dseg(rl_msgs::seg_scene_srv::Request  &req, 
                             rl_msgs::seg_scene_srv::Response &res)
{

    ROS_INFO_STREAM("Get Input");
    PointCloud<PointXYZRGB>::Ptr cloud_input(new PointCloud<PointXYZRGB>);
    sensor_msgs::Image msg_img;
    if( req.use_camera )
    {
        ROS_INFO_STREAM("Get Input from Camera");
        // Get Point Cloud        
        if( !GetPointCloudFromCamera(cloud_input) ) return false;
        // Get Image        
        if( !GetImageFromCamera(msg_img) ) return false;
    }
    else
    {
        ROS_INFO_STREAM("Get Input from Arg");

        // Get Point Cloud
        fromROSMsg<PointXYZRGB>(req.cloud, *cloud_input);
        // Get Image                
        msg_img = req.image;
    }
    ROS_INFO_STREAM("Get Input - Done");

    // Get Intrinsic Camera Parameters    
    if( req.use_camera )
    {        
        if( req.camera_K.size()==0 )
        {
            sensor_msgs::CameraInfo::ConstPtr ci_depth 
             = ros::topic::waitForMessage<sensor_msgs::CameraInfo>
                 (SUB_CAMINFO,*nh,ros::Duration(1));
            for( int k=0; k<9; k++ ) camera_K[k] = ci_depth->K[k];
        }
        else
        {
            for( int k=0; k<9; k++ ) camera_K[k] = req.camera_K[k];
        }        
    }
    else
    {
        if( req.camera_K.size()>0 )        
        {
            for( int k=0; k<9; k++ ) camera_K[k] = req.camera_K[k];
        }
    }

    if( req.camera_RT.size()>0 )
    {
        for( int rt=0; rt<12; rt++ ) camera_RT[rt] = req.camera_RT[rt];        
    }

    return seglccp2dseg->Segment( &res.segscene,
                                  msg_img, cloud_input, 
                                  req.param, camera_K, camera_RT );
}

#if 0
bool Segmentation_lccp(segmentation::seg_scene_srv::Request  &req, 
                       segmentation::seg_scene_srv::Response &res)
{
    SuperVoxelParam param_sv;
    param_sv.parse_param(req.param);
    LCCPParam param_lccp;
    param_lccp.parse_param(req.param);

    // Get Point Cloud
    ROS_INFO_STREAM("Get Input");
    PointCloud<PointXYZRGB>::Ptr cloud_input(new PointCloud<PointXYZRGB>);
    if( !GetPointCloudFromCamera(cloud_input) ) return false;
    ROS_INFO_STREAM("Get Input - Done");

    ROS_INFO_STREAM("Remove Background/Noise");
    vector<int> index;
    removeNaNFromPointCloud(*cloud_input,*cloud_input,index);        
    RemoveBackground(cloud_input);
    ROS_INFO_STREAM("Remove Background/Noise - Done");
    if( cloud_input->size() <= 0 )
    {
        ROS_WARN_STREAM("Too small # of point clouds");
        res.supervoxels.clear();
        return true;
    }

    /// Preparation of Input: Supervoxel Oversegmentation
    ROS_INFO_STREAM("Supervoxels");
    multimap<uint32_t, uint32_t> supervoxel_adjacency;
    map<uint32_t, Supervoxel<PointXYZRGB>::Ptr> supervoxel_clusters;
    PointCloud<PointXYZL>::Ptr sv_labeled_cloud(new PointCloud<PointXYZL>);
    DoSupervoxelClustering( cloud_input, param_sv, 
                            &supervoxel_clusters, 
                            &supervoxel_adjacency,
                            &sv_labeled_cloud );
    ROS_INFO_STREAM("Supervoxels - Done");

    /// The Main Step: Perform LCCPSegmentation
    ROS_INFO_STREAM("Segmentation");
    PointCloud<PointXYZL>::Ptr lccp_labeled_cloud;
    DoSegmentation( supervoxel_clusters, 
                    supervoxel_adjacency,
                    param_lccp,
                    sv_labeled_cloud,
                    &lccp_labeled_cloud            );
    ROS_INFO_STREAM("Segmentation - Done");

    // service & publish
    ROS_INFO_STREAM("PointXYZL -> Supervoxel");

    map<uint32_t, Supervoxel<PointXYZRGB>::Ptr> supervoxel_out;
    for(int p=0; lccp_labeled_cloud->points.size(); p++ )
    {
        PointXYZL &ptl = lccp_labeled_cloud->points[p];
        map<uint32_t, Supervoxel<PointXYZRGB>::Ptr>::iterator it_l2sv
         = supervoxel_out.find(ptl.label);

        Supervoxel<PointXYZRGB>::Ptr sv;
        if( it_l2sv == supervoxel_out.end() )
        {
            sv = Supervoxel<PointXYZRGB>::Ptr(new Supervoxel<PointXYZRGB>);
            supervoxel_out.insert(
              pair<uint32_t,Supervoxel<PointXYZRGB>::Ptr>(ptl.label,sv));
        }
        else
        {
            sv = it_l2sv->second;
        }
        PointXYZRGB pt;
        pt.x=ptl.x; pt.y=ptl.y; pt.z=ptl.z;

        sv->voxels_->push_back(pt);
    }
    ROS_INFO_STREAM("PointXYZL -> Supervoxel - Done");

    toROSMsg(supervoxel_out, res.supervoxels);

    return true;
}
#endif

int main(int argc, char* argv[])
{
    // ROS
    ros::init(argc,argv,"segmentation_node");
    nh = new ros::NodeHandle;
    ParseParam(*nh);

#if 0
    ros::ServiceServer srv_seg
     = nh->advertiseService(SRV_LCCP, Segmentation_lccp);
#endif
    ros::ServiceServer srv_seg_2Dseg
     = nh->advertiseService(SRV_LCCP_2DSEG, Segmentation_lccp_2Dseg);

    seglccp2dseg = new SegLCCP2DSeg(nh, SUB_COB);

    ros::spin();

    delete seglccp2dseg;
    delete nh;

    return 0;
}
