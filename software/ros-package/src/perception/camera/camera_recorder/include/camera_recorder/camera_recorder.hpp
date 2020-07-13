#ifndef CAMERA_RECORDER__H__
#define CAMERA_RECORDER__H__

#include <thread>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <sensor_msgs/Image.h>

typedef class CameraRecorder
{
public:
    CameraRecorder( const ros::NodeHandle &nh,
                    const std::string &dir_save=getenv("HOME"), 
                    const std::string &prefix_save="",
                    const std::string &topic_cloud="/pointcloud",
                    const std::string &topic_image="/rgb/image",
                    const std::string &topic_depth="/depth/image" );
    ~CameraRecorder(){};

    void SaveCurrentFrame();
    void ResetCounter(){ idx_save = 0; }
    void SetPrefix(std::string prefix){ prefix_save = prefix; }
    void SetSaveDir(std::string dir){ dir_save = dir; }

    void StartSave(double sec_duration);
    void StopSave();
private:

    bool GetPointCloudFromCamera(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
    bool GetImageFromCamera( cv::Mat &image );
    bool GetDepthFromCamera( cv::Mat &depth );

    void ThreadSave(double sec_duration);
    bool running_save;
    std::thread* t_save;
    
    ros::NodeHandle nh;

    std::string dir_save;
    std::string prefix_save;
    std::string topic_cloud;
    std::string topic_image;
    std::string topic_depth;

    int idx_save;
} CameraRecorder;

#endif