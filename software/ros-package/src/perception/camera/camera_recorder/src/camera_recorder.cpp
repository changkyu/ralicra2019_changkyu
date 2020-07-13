#include <ctime>
#include <chrono>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>

#include <sensor_msgs/PointCloud2.h>

#include "camera_recorder/camera_recorder.hpp"

using namespace std;
using namespace pcl;

CameraRecorder::CameraRecorder( const ros::NodeHandle &nh_in,
                                const std::string &dir_save_in, 
                                const std::string &prefix_save_in,
                                const std::string &topic_cloud_in,
                                const std::string &topic_image_in,
                                const std::string &topic_depth_in   )
{
    idx_save = 0;
    running_save = false;
    t_save = NULL;

    nh = nh_in;
    dir_save    = dir_save_in;
    prefix_save = prefix_save_in;
    topic_cloud = topic_cloud_in;
    topic_image = topic_image_in;
    topic_depth = topic_depth_in;
}

bool CameraRecorder::GetPointCloudFromCamera(PointCloud<PointXYZRGB>::Ptr cloud)
{        
    try
    {
        sensor_msgs::PointCloud2::ConstPtr msg_pc
         = ros::topic::waitForMessage<sensor_msgs::PointCloud2>
             (topic_cloud, nh, ros::Duration(6.0));
        fromROSMsg<pcl::PointXYZRGB>(*msg_pc, *cloud);
    }
    catch(std::exception &e)
    {
        ROS_ERROR("Exception during waitForMessage PointCloud2: %s", e.what());
        return false;
    }
    return true;
}

bool CameraRecorder::GetImageFromCamera( cv::Mat &image )
{
    try
    {
        sensor_msgs::Image::ConstPtr msg_ptr;
        msg_ptr = ros::topic::waitForMessage<sensor_msgs::Image>
                    (topic_image, nh, ros::Duration(6.0));
        cv_bridge::CvImagePtr cv_ptr_image
         = cv_bridge::toCvCopy(msg_ptr,sensor_msgs::image_encodings::BGR8);

        cv_ptr_image->image.copyTo(image);
    }
    catch(std::exception &e)
    {
        ROS_ERROR("Exception during waitForMessage Image: %s", e.what());
        return false;
    }
    return true;
}

bool CameraRecorder::GetDepthFromCamera( cv::Mat &depth )
{
    try
    {
        sensor_msgs::Image::ConstPtr msg_ptr;
        msg_ptr = ros::topic::waitForMessage<sensor_msgs::Image>
                    (topic_depth, nh, ros::Duration(6.0));

        cv_bridge::CvImagePtr cv_ptr_depth
         = cv_bridge::toCvCopy(msg_ptr,sensor_msgs::image_encodings::TYPE_16UC1);
        cv_ptr_depth->image.copyTo(depth);
    }
    catch(std::exception &e)
    {
        ROS_ERROR("Exception during waitForMessage Image: %s", e.what());
        return false;
    }
    return true;
}

void CameraRecorder::SaveCurrentFrame()
{
    char name[256];
    sprintf(name,"%s.%06d",prefix_save.c_str(),idx_save); 
    
    cv::Mat image, depth;
    GetImageFromCamera(image);
    GetDepthFromCamera(depth);
    PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
    GetPointCloudFromCamera(cloud);

    // Image
    imwrite(dir_save + "/" + name + ".color.png", image);
    
    // depth
    imwrite(dir_save + "/" + name + ".depth.png", depth);
    
    // point
    io::savePLYFileBinary(dir_save + "/" + name + ".cloud.ply", *cloud);
    io::savePCDFileBinary(dir_save + "/" + name + ".cloud.pcd", *cloud);
    
    ROS_INFO_STREAM("Saved " << dir_save << "/" << name);
    
    idx_save++;
}

void CameraRecorder::ThreadSave(double sec_duration)
{
    while(running_save)
    {
        clock_t begin = clock();

        SaveCurrentFrame();

        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

        std::this_thread::sleep_for(
            std::chrono::milliseconds((long int)(sec_duration-elapsed_secs)/10));
    }
}

void CameraRecorder::StartSave(double sec_duration)
{
    StopSave();

    running_save = true;
    t_save = new thread(&CameraRecorder::ThreadSave, this, sec_duration);
}

void CameraRecorder::StopSave()
{
    running_save = false;
    if( t_save )
    {        
        t_save->join();
        delete t_save;
        t_save = NULL;
    }
}