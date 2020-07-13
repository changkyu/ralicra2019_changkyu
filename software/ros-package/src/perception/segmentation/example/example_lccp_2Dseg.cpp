#include <sstream>
#include <map>

#include <boost/program_options.hpp>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/highgui.hpp>

#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter.h>

#include "rl_msgs/seg_scene_srv.h"
#include "rl_msgs/rl_msgs_visualization.hpp"

using namespace std;
using namespace cv;
using namespace pcl;

namespace po = boost::program_options;
ros::ServiceClient clt_lccp_2Dseg;
bool save_snapshot = false;
bool remove_background = true;
bool vis=true;
string name_2dseg="cob";
string param="";

template<typename PointT>
void VisSupervoxel(
  pcl::visualization::PCLVisualizer &viewer,
  map<uint32_t, typename Supervoxel<PointT>::Ptr> &supervoxel_clusters
  //multimap<uint32_t, uint32_t> &supervoxel_adjacency            )
  )
{
    viewer.removeAllPointClouds();
    for(typename map<uint32_t,typename Supervoxel<PointT>::Ptr>::iterator
          it_map = supervoxel_clusters.begin();
        it_map != supervoxel_clusters.end(); 
        it_map++                                          )
    {   
        PointCloud<PointXYZL>::Ptr cloud(new PointCloud<PointXYZL>);
        
        for(typename PointCloud<PointT>::iterator it_pt=it_map->second->voxels_->begin(); 
            it_pt!=it_map->second->voxels_->end(); it_pt++)
        {
            PointXYZL pt;
            pt.x = it_pt->x;
            pt.y = it_pt->y;
            pt.z = it_pt->z;
            pt.label = it_map->first;
            cloud->push_back(pt);
        }

        stringstream ss;
        ss << it_map->first;
        viewer.addPointCloud(cloud, ss.str() );        
    }
}

void fromROSMsg(vector<rl_msgs::SegmentationObject> &vec_rlsv,
                map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr> &map_sv
)
{
    for( int i=0; i<vec_rlsv.size(); i++ )
    {
        pcl::Supervoxel<pcl::PointXYZRGB>::Ptr 
          sv(new pcl::Supervoxel<pcl::PointXYZRGB>);        
        pcl::fromROSMsg(vec_rlsv[i].cloud, *sv->voxels_);        
        map_sv.insert( 
            pair<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr>(i,sv) );
    }
}

void Segmentation_lccp_2Dseg( 
    map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr> *l2sv)
{
    rl_msgs::seg_scene_srv srv;
    srv.request.use_camera = true;
    
    save_snapshot ? srv.request.param += ",save_snapshot=1":srv.request.param += ",save_snapshot=0";
    remove_background ? srv.request.param+=",remove_background=1":srv.request.param+=",remove_background=0";
    if( param.compare("")!=0 ) srv.request.param += ("," + param);
    srv.request.param += ",name_2dseg=" + name_2dseg;

    if( clt_lccp_2Dseg.call(srv) )
    {
        fromROSMsg(srv.response.segscene.objects,*l2sv);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to do segmentation [lccp_2Dseg]");
    }
}

void Segmentation_lccp_2Dseg( 
    visualization::PCLVisualizer::Ptr viewer )
{
    rl_msgs::seg_scene_srv srv;
    srv.request.use_camera = true;

    srv.request.param = param;
    save_snapshot ? srv.request.param += ",save_snapshot=1":srv.request.param += ",save_snapshot=0";
    remove_background ? srv.request.param+=",remove_background=1":srv.request.param+=",remove_background=0";
    if( param.compare("")!=0 ) srv.request.param += ("," + param);    
    srv.request.param += ",name_2dseg=" + name_2dseg;

    if( clt_lccp_2Dseg.call(srv) )
    {
        if(viewer)
        {
            int v0(0), v1(0);
            viewer->createViewPort (0.0, 0.0, 0.5, 1.0, v0);
            viewer->createViewPort (0.5, 0.0, 1.0, 1.0, v1);
            viewer->removeAllPointClouds();
            viewer->removeAllShapes();
            AddSegmentationObjects(srv.response.segscene.objects,*viewer, v0);
            AddSegmentationObjects2(srv.response.segscene.objects,*viewer, v1);            
        } 
    }
    else
    {
        ROS_ERROR_STREAM("Failed to do segmentation [lccp_2Dseg]");
    }
}

void Segmentation_lccp_2Dseg(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, Mat &image,
    vector<float> &camera_K,
    map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr> *l2sv )
{
    rl_msgs::seg_scene_srv srv;

    srv.request.use_camera = false;

    srv.request.param = param;
    save_snapshot ? srv.request.param += ",save_snapshot=1":srv.request.param += ",save_snapshot=0";
    remove_background ? srv.request.param+=",remove_background=1":srv.request.param+=",remove_background=0";
    if( param.compare("")!=0 ) srv.request.param += ("," + param);    
    srv.request.param += ",name_2dseg=" + name_2dseg;

    srv.request.camera_K = camera_K;
    pcl::toROSMsg(*cloud,srv.request.cloud);
        
    cv_bridge::CvImage msg_image;    
    msg_image.encoding = sensor_msgs::image_encodings::BGR8;
    msg_image.image    = image;
    msg_image.toImageMsg(srv.request.image);
        
    if( clt_lccp_2Dseg.call(srv) )
    {
        fromROSMsg(srv.response.segscene.objects,*l2sv);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to do segmentation [lccp_2Dseg]");
    }
}

void Segmentation_lccp_2Dseg(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, Mat &image,
    vector<float> &camera_K,
    visualization::PCLVisualizer::Ptr viewer )
{
    rl_msgs::seg_scene_srv srv;

    srv.request.use_camera = false;

    srv.request.param = param;
    save_snapshot ? srv.request.param += ",save_snapshot=1":srv.request.param += ",save_snapshot=0";
    remove_background ? srv.request.param+=",remove_background=1":srv.request.param+=",remove_background=0";
    if( param.compare("")==0 ) srv.request.param += ("," + param);    
    srv.request.param += ",name_2dseg=" + name_2dseg;

    srv.request.camera_K = camera_K;
    pcl::toROSMsg(*cloud,srv.request.cloud);
        
    cv_bridge::CvImage msg_image;    
    msg_image.encoding = sensor_msgs::image_encodings::BGR8;
    msg_image.image    = image;
    msg_image.toImageMsg(srv.request.image);
        
    if( clt_lccp_2Dseg.call(srv) )
    {
        if(viewer)
        {
            int v0(0), v1(0);
            viewer->createViewPort (0.0, 0.0, 0.5, 1.0, v0);
            viewer->createViewPort (0.5, 0.0, 1.0, 1.0, v1);
            viewer->removeAllPointClouds();
            viewer->removeAllShapes();
            AddSegmentationObjects(srv.response.segscene.objects,*viewer, v0);
            AddSegmentationObjects2(srv.response.segscene.objects,*viewer, v1);            
        } 
    }
    else
    {
        ROS_ERROR_STREAM("Failed to do segmentation [lccp_2Dseg]");
    }
}

bool flag = true;
pcl::visualization::PCLVisualizer::Ptr viewer_pts;
pcl::visualization::PCLVisualizer::Ptr viewer_seg;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
void Callback_pclkeyboard (const pcl::visualization::KeyboardEvent &event,
                           void* viewer_void)
{
    if (event.keyDown())
    {
        if( event.getKeySym()=="Escape" ||
            event.getKeyCode()=='Q'     ||
            event.getKeyCode()=='q'        )
        {
            flag = false;
        }
        else if( event.getKeySym()=="space" )
        {            
            Segmentation_lccp_2Dseg(viewer_seg);
        }        
    }
}

void Callback_point(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    static bool busy=false;
    if( busy ) return;
    pcl::fromROSMsg<pcl::PointXYZRGB>(*msg, *cloud);
    std::vector<int> index;
    pcl::removeNaNFromPointCloud(*cloud,*cloud,index);
    if(cloud->size() > 0)
    {
        busy=true;        
        viewer_pts->updatePointCloud(cloud);
        viewer_pts->spinOnce(10);
        busy=false;
    }
}

#define ROS_TOPIC_POINT ("/pointcloud")
int main(int argc, char* argv[])
{
    string fp_image;
    string fp_cloud;    
    
    po::options_description desc("Example Usage");
    desc.add_options()
        ("help", "help")
        ("image,i", po::value<string>(&fp_image),    "input image")
        ("cloud,c", po::value<string>(&fp_cloud),    "input cloud")
        ("save,s",  po::value<bool>(&save_snapshot)->default_value(false), "Save Snapshot?")
        ("background,b",  po::value<bool>(&remove_background)->default_value(true), "Remove Background?")
        ("method,m",po::value<string>(&name_2dseg)->default_value("quickshift"),    "2dseg name")
        ("vis,v",   po::value<bool>(&vis)->default_value(true), "Visualization")
        ("param,p", po::value<string>(&param)->default_value(""), "Other param to pass")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) 
    {
        cout << desc << "\n";
        return 0;
    }

    // ROS init
    ros::init(argc,argv,"example_segmentation_lccp");
    ros::NodeHandle nh;      
    clt_lccp_2Dseg = nh.serviceClient<rl_msgs::seg_scene_srv>
                                                    ("segmentation/lccp_2Dseg");

    if( vm.count("image") ) // Using pre-obtained image and pointcloud
    {
        // Get input
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr
         cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (fp_cloud, *cloud) == -1)
        {
            ROS_ERROR_STREAM ("Couldn't read file: " << fp_cloud);
            return (-1);
        }
        Mat img = imread(fp_image);

        vector<float> K = {615.957763671875,           0.0, 308.10989379882810, 
                                    0.0, 615.9578247070312, 246.33352661132812, 
                                    0.0,               0.0,               1.0 };

        map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr> l2sv;

        if( vis )
        {
            viewer_seg = pcl::visualization::PCLVisualizer::Ptr(
                            new pcl::visualization::PCLVisualizer("Segmentation Result"));
            viewer_seg->setSize(600,480);
            viewer_seg->setPosition(600,0);        
            //viewer_seg->registerKeyboardCallback(Callback_pclkeyboard);
            viewer_seg->setCameraPosition(0,0,-1,0,0,1,0,-1,0);
        
            Segmentation_lccp_2Dseg(cloud, img, K, viewer_seg);

            viewer_seg->spin();
            viewer_seg->close();
        }
        else
        {            
            Segmentation_lccp_2Dseg(cloud, img, K, NULL);
        }
    }
    else
    {
        ros::Subscriber sub_point;    
        sub_point = nh.subscribe(ROS_TOPIC_POINT,10,Callback_point);

        viewer_seg = pcl::visualization::PCLVisualizer::Ptr(
                        new pcl::visualization::PCLVisualizer("Segmentation Result"));
        viewer_seg->setSize(600,480);
        viewer_seg->setPosition(600,0);
        viewer_seg->addPointCloud(cloud);
        viewer_seg->registerKeyboardCallback(Callback_pclkeyboard);
        viewer_seg->setCameraPosition(0,0,-1,0,0,1,0,-1,0);
    
        viewer_pts = pcl::visualization::PCLVisualizer::Ptr(
                        new pcl::visualization::PCLVisualizer("Input Cloud"));
        viewer_pts->setSize(600,480);
        viewer_pts->setPosition(0,0);
        viewer_pts->addPointCloud(cloud);
        viewer_pts->registerKeyboardCallback(Callback_pclkeyboard);
        viewer_pts->setCameraPosition(0,0,-1,0,0,1,0,-1,0);

        ros::Rate r(30);
        while(ros::ok() && flag)
        {
            viewer_pts->spinOnce(10);
            viewer_seg->spinOnce(10);
            ros::spinOnce();            
        }
        ros::shutdown();

        viewer_pts->close();
        viewer_seg->close();    
    }
    return 0;
}