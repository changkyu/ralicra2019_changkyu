#include <boost/program_options.hpp>

#include <opencv2/highgui.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>

#include <sensor_msgs/CameraInfo.h>
#include "rl_msgs/seg_scene_srv.h"
#include "rl_msgs/tracking_cloud_srv.h"
#include "rl_msgs/rl_msgs_visualization.hpp"

#include "tracking/cloud_tracker.hpp"

using namespace std;
using namespace cv;
using namespace pcl;

namespace po = boost::program_options;

ros::ServiceClient clt_cloud_tracker;
ros::ServiceClient clt_lccp_2Dseg;

bool flag = true;

void Callback_pclkeyboard (const pcl::visualization::KeyboardEvent &event,
                           void* viewer_void)
{
    if (event.keyDown())
    {
        if( event.getKeySym()=="Escape" ||
            event.getKeyCode()=='Q'     ||
            event.getKeyCode()=='q'         )
        {
            flag = false;
        }        
    }
}

int main(int argc, char* argv[])
{
    float K[9] = {615.957763671875,               0.0, 308.10989379882810, 
                               0.0, 615.9578247070312, 246.33352661132812, 
                               0.0,               0.0,               1.0 };
    vector<float> camera_K(9);
    for( int i=0; i<9; i++ ) camera_K[i] = K[i];

    string dp_image;
    string prefix;
    vector<int> range_index;
    
    po::options_description desc("Example Usage");
    desc.add_options()
        ("help", "help")
        ("inputdir,i",   po::value<string>(&dp_image),
                         "input image director")        
        ("prefix,p",     po::value<string>(&prefix)
                         ->default_value(""),
                         "input image filename format"),
        ("indexrange,r", po::value<vector<int> >(&range_index)->multitoken(),
                         "input image index range (0,n)")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if( vm.count("help")        ||
        dp_image.compare("")==0 || 
        range_index.size()!=2      ) 
    {
        cout << dp_image << endl;
        cout << fmt_image << endl;
        cout << range_index.size() << endl;
        cout << desc << "\n";
        return 0;
    }

    // Read inputs
    const int n_images = range_index[1] - range_index[0] + 1;
    vector<Mat> images(n_images);
    vector<Mat> depths(n_images);
    vector<PointCloud<PointXYZRGB>::Ptr> clouds(n_images);
    for( int i=0; i<n_images; i++ )
    {        
        char fp_image[256];
        sprintf(fp_image,(dp_image + "/" + "%s%06d.color.png").c_str(),
                prefix.c_str(),i+range_index[0]);
        cout << fp_image << endl;
        images[i] = imread(fp_image);

        char fp_depth[256];
        sprintf(fp_depth,(dp_image + "/" + "%s%06d.depth.png").c_str(),
                prefix.c_str(),i+range_index[0]);
        cout << fp_depth << endl;
        depths[i] = imread(fp_depth,CV_LOAD_IMAGE_ANYDEPTH);

        char fp_cloud[256];
        sprintf(fp_cloud,(dp_image + "/" + "%s%06d.cloud.pcd").c_str(),
                prefix.c_str(),i+range_index[0]);        
        cout << fp_cloud << endl;

        clouds[i] = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
        if (io::loadPCDFile<PointXYZRGB> (fp_cloud, *clouds[i]) == -1)
        {
            ROS_ERROR_STREAM ("Couldn't read file: " << fp_cloud);
            return (-1);
        }
    }

    ros::init(argc,argv,"example_cloud_tracker");
    ros::NodeHandle nh;
    CloudTracker _cloudTracker(&nh, camera_K, images[0].cols, images[0].rows);
    
    vector<CloudTracker::Matrix4> TFs;
    _cloudTracker.Track(images, depths, clouds, TFs);
    for( size_t s=0; s<TFs.size(); s++ )
    {
        cout << TFs[s] << endl;
    }
}
