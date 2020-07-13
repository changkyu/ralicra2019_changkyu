#include <iostream>
#include <boost/filesystem.hpp>

//////////////////////////////////////////////////////////////////////////////
// ROS
//////////////////////////////////////////////////////////////////////////////
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>

//////////////////////////////////////////////////////////////////////////////
// OPENCV
//////////////////////////////////////////////////////////////////////////////
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

//////////////////////////////////////////////////////////////////////////////
// TF
//////////////////////////////////////////////////////////////////////////////
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

//////////////////////////////////////////////////////////////////////////////
// PCL
//////////////////////////////////////////////////////////////////////////////
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>

namespace fs = boost::filesystem;

using namespace std;
using namespace pcl;
using namespace cv;


bool running = true;

string str_outdir;
visualization::PCLVisualizer::Ptr viewer_pts;
cv_bridge::CvImagePtr cv_ptr_image;
cv_bridge::CvImagePtr cv_ptr_depth;
PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);

sensor_msgs::CameraInfo::ConstPtr ci_depth;

string ROS_TOPIC_IMAGE;
string ROS_TOPIC_DEPTH;
string ROS_TOPIC_POINT;
bool show_image;
bool show_depth;
bool show_point;
void ParseParam(ros::NodeHandle &nh)
{
    nh.param<string>("outdir",str_outdir,getenv("HOME"));
    ROS_INFO("outdir: %s",str_outdir.c_str());
   
    nh.param<bool>("show_image",show_image,true);
    nh.param<bool>("show_depth",show_depth,true);
    nh.param<bool>("show_point",show_point,true);

    nh.param<string>("topic_image",ROS_TOPIC_IMAGE,
                            "rgb/image");

    nh.param<string>("topic_depth",ROS_TOPIC_DEPTH,
                            "/depth/image");

    nh.param<string>("topic_point",ROS_TOPIC_POINT,
                            "/pointcloud");
}

void SaveAll()
{
/*
    fs::path outdir(str_outdir);
    if(!fs::is_directory(outdir))
    {
        fs::create_directories(str_outdir);
    }
*/
    static int idx=0;
    char name[256];
    sprintf(name,"%06d",idx); 

    // Image
    if(show_image)
    {
        imwrite(str_outdir + "/" + name + ".color.png", cv_ptr_image->image);
    }

    // depth
    if(show_depth)
    {
        Point minLoc,maxLoc;
        double minVal, maxVal;
        minMaxLoc(cv_ptr_depth->image, &minVal, &maxVal, &minLoc, &maxLoc);
        ROS_INFO_STREAM("min: " << minVal << ", max: "<< maxVal);

        imwrite(str_outdir + "/" + name + ".depth.png", cv_ptr_depth->image);
    }

    // point
    if(show_point)
    {
        io::savePLYFileBinary(str_outdir + "/" + name + ".ply", *cloud);
        io::savePCDFileBinary(str_outdir + "/" + name + ".pcd", *cloud);
    }

    ROS_INFO_STREAM("Saved " << str_outdir << "/" << name);
    idx++;
}

void Callback_pclkeyboard (const visualization::KeyboardEvent &event,
                           void* viewer_void)
{
    visualization::PCLVisualizer *viewer
     = static_cast<visualization::PCLVisualizer *> (viewer_void);
    if (event.keyDown())
    {
        if( event.getKeySym()=="Escape" )
        {
            running = false;            
        }
        else
        {
            SaveAll();
        }
    }
}

void Callback_cvkeyboard( int key )
{
    if( key != -1 )
    {
        switch(key%256)
        {
            case 27: // ESC
                running = false;
            break;
            default:
                SaveAll();
            break;
        }
    }
}

void Callback_image(const sensor_msgs::Image::ConstPtr& msg)
{
    try
    {
        cv_ptr_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    imshow("image", cv_ptr_image->image);    
}

void Callback_depth(const sensor_msgs::Image::ConstPtr& msg)
{
    Point minLoc,maxLoc;
    double minVal, maxVal;
    try
    {
        cv_ptr_depth = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        minMaxLoc(cv_ptr_depth->image, &minVal, &maxVal, &minLoc, &maxLoc);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    Mat depth;
    cv_ptr_depth->image.convertTo(depth,CV_32FC1);
    imshow("depth", (depth-minVal)/maxVal);    
}

void Callback_point(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    static bool busy=false;
    if( busy ) return;

    fromROSMsg<PointXYZRGB>(*msg, *cloud);
    vector<int> index;
    removeNaNFromPointCloud(*cloud,*cloud,index);
    if(cloud->size() > 0)
    {
        busy=true;        
        viewer_pts->updatePointCloud(cloud);
        if(show_point) viewer_pts->spinOnce(10);
        busy=false;
    }
}

int main(int argc, char* argv[])
{
    // ROS
    ros::init(argc,argv,"livecam");
    ros::NodeHandle nh;

    ParseParam(nh);

    // ROS - subscribers
    ros::Subscriber sub_image;
    if(show_image)
    {
        sub_image = nh.subscribe(ROS_TOPIC_IMAGE,10,Callback_image);
        ROS_INFO_STREAM("sub: " << ROS_TOPIC_IMAGE);
        namedWindow("image");
        moveWindow("image", 0,600);
    }
    ros::Subscriber sub_depth;
    if(show_depth)
    {
        sub_depth = nh.subscribe(ROS_TOPIC_DEPTH,10,Callback_depth);
        ci_depth = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/depth/camera_info",nh,ros::Duration(0));
        ROS_INFO_STREAM("ci_depth: " << endl << *ci_depth);
        namedWindow("depth");
        moveWindow("depth", 700,600);
    }
    ros::Subscriber sub_point;
    if(show_point)
    {
        sub_point = nh.subscribe(ROS_TOPIC_POINT,10,Callback_point);
    }

    // PCL
    if(show_point)
    {
        viewer_pts = visualization::PCLVisualizer::Ptr(
          new visualization::PCLVisualizer("Point Cloud (Input)"));
        viewer_pts->setPosition(0,0);
        viewer_pts->setSize(600,480);
        viewer_pts->setCameraPosition(0,0,-1,0,0,1,0,-1,0);
        viewer_pts->addPointCloud(cloud);
        viewer_pts->registerKeyboardCallback(Callback_pclkeyboard, (void*)&viewer_pts);
    }

    ros::Rate r(30);    
    while( ros::ok() && running )
    {
        ros::spinOnce();
        viewer_pts->spinOnce(1);
        Callback_cvkeyboard(waitKey(1));

        //r.sleep();
    }
    ros::shutdown();

    return 0;
}
