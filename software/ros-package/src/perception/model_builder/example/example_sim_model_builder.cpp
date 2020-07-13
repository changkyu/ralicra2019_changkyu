#include <iostream>

#include <boost/program_options.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>

#include "model_builder/vis_model_builder.hpp"
#include "model_builder/sim_model_builder.hpp"

using namespace std;
using namespace cv;
using namespace pcl;

namespace po = boost::program_options;

Eigen::Matrix4f tf_cam2world;

vector<float> camera_K;
vector<float> camera_RT;

void ParseParam(ros::NodeHandle nh)
{
    nh.getParam("/segmentation/camera_info/intrinsic", camera_K);
    nh.getParam("/segmentation/camera_info/extrinsic", camera_RT);

    for( int i=0; i<16; i++ ) tf_cam2world(i/4,i%4) = camera_RT[i];
}

int main(int argc, char* argv[])
{
    string dp_image;
    string fmt_image;
    string prefix_image;
    vector<int> range_index;
    
    po::options_description desc("Example Usage");
    desc.add_options()
        ("help", "help")
        ("inputdir,i",   po::value<string>(&dp_image),
                         "input image director")        
        ("prefix,p",     po::value<string>(&prefix_image)
                         ->default_value("scene%d"),
                         "input image filename format")
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
        cout << prefix_image << endl;
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
                prefix_image.c_str(),i+range_index[0]);
        cout << fp_image << endl;
        images[i] = imread(fp_image);

        char fp_depth[256];
        sprintf(fp_depth,(dp_image + "/" + "%s%06d.depth.png").c_str(),
                prefix_image.c_str(),i+range_index[0]);
        cout << fp_depth << endl;
        depths[i] = imread(fp_depth,CV_LOAD_IMAGE_ANYDEPTH);

        char fp_cloud[256];
        sprintf(fp_cloud,(dp_image + "/" + "%s%06d.cloud.pcd").c_str(),
                prefix_image.c_str(),i+range_index[0]);
        cout << fp_cloud << endl;
        clouds[i] = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
        if (io::loadPCDFile<PointXYZRGB> (fp_cloud, *clouds[i]) == -1)
        {
            ROS_ERROR_STREAM ("Couldn't read file: " << fp_cloud);
            return (-1);
        }
    }

    ros::init(argc,argv,"example_vis_model_builder");
    ros::NodeHandle nh;
    ParseParam(nh);    
    VisModelBuilder _visModelBuilder(&nh, camera_K, camera_RT, "name_2dseg=quickshift", images[0].cols, images[0].rows);
#if 0
    VisModelBuilder::Plane plane;
    plane.coef[0]=0; plane.coef[1]=0; plane.coef[2]=1; plane.coef[3]=-0.46;
    _visModelBuilder.AddBGPlane(plane);
    plane.coef[0]=-1; plane.coef[1]=0; plane.coef[2]=0; plane.coef[3]=-1.08;
    _visModelBuilder.AddBGPlane(plane);
#endif 

    const int beg=0, end=images.size()-1; 

    ROS_INFO_STREAM("first update");
    _visModelBuilder.Update(images[0],depths[0],clouds[0]);
    ROS_INFO_STREAM("first update - Done");
    for( size_t i=1; i<=end; i++ )
    {        
        ROS_INFO_STREAM( i << "-th update");
        if( i<end )
        {
            _visModelBuilder.Update(images[i],depths[i]);
        }
        else
        {
            _visModelBuilder.Update(images[i],depths[i],clouds[i]);
        }
        ROS_INFO_STREAM( i << "-th update - Done");
    }

    ROS_INFO_STREAM( "build model");
    size_t n_models = _visModelBuilder.NumOfModels();
    vector<VisModelBuilder::VisConvexHull> chulls(n_models);
    for( size_t o=0; o<n_models; o++ )
    {
        _visModelBuilder.GenerateModel(o,chulls[o]);
    }    
    ROS_INFO_STREAM( "build model - Done");

    SimModelBuilder _simModelBuilder;
    vector<VisModelBuilder::Plane> &planes = _visModelBuilder.GetBGPlanes();
    for( size_t pl=0; pl<planes.size(); pl++ )
    {
        _simModelBuilder.AddStaticVisModel(planes[pl]);
    }    
    _simModelBuilder.AddInputVisModels(chulls);
    _simModelBuilder.TestVisModels();
}
