#include <boost/program_options.hpp>

#include <opencv2/highgui.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl/surface/convex_hull.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>

#include <sensor_msgs/CameraInfo.h>
#include "rl_msgs/seg_scene_srv.h"

#include "model_builder/vis_model_builder.hpp"

using namespace std;
using namespace cv;
using namespace pcl;

namespace po = boost::program_options;

bool flag = true;

vector<float> camera_K;
vector<float> camera_RT;

void ParseParam(ros::NodeHandle nh)
{
    nh.getParam("/segmentation/camera_info/intrinsic", camera_K);
    nh.getParam("/segmentation/camera_info/extrinsic", camera_RT);
}

void Callback_pclkeyboard (const visualization::KeyboardEvent &event,
                           void* viewer_void)
{
    if (event.keyDown())
    {
        string keySym  = event.getKeySym();
        int keyCode = event.getKeyCode();
        if( keySym=="Escape" ||
            keyCode=='Q'     ||
            keyCode=='q'         )
        {
            flag = false;
        }        
    }
}

int main(int argc, char* argv[])
{
    string dp_image;
    string fmt_image;
    string prefix_image;
    vector<int> range_index;
    bool remove_background;
    
    po::options_description desc("Example Usage");
    desc.add_options()
        ("help", "help")
        ("inputdir,i",   po::value<string>(&dp_image),
                         "input image director")
        ("prefix,p",     po::value<string>(&prefix_image)
                         ->default_value("scene%d"),
                         "input image filename format")
        ("inputfmt,f",   po::value<string>(&fmt_image)
                         ->default_value("%06d.color.png"),
                         "input image filename format")
        ("background,b", po::value<bool>(&remove_background)
                         ->default_value(true),
                         "remove_background?")
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

    stringstream ss;    
    if( remove_background ) ss << ",remove_background=1";
    else                    ss << ",remove_background=0";

    VisModelBuilder _visModelBuilder(&nh, camera_K, camera_RT, 
                                     "name_2dseg=quickshift"+ss.str(), 
                                     images[0].cols, images[0].rows);
    
    VisModelBuilder::Plane plane;
    plane.coef[0]=0; plane.coef[1]=0; plane.coef[2]=1; plane.coef[3]=-0.46;
    _visModelBuilder.AddBGPlane(plane);
    plane.coef[0]=-1; plane.coef[1]=0; plane.coef[2]=0; plane.coef[3]=-1.08;
    _visModelBuilder.AddBGPlane(plane);

    const int beg=0, end=images.size()-1; 
    _visModelBuilder.Update(images[0],depths[0],clouds[0]);
    for( size_t i=1; i<=end; i++ )
    {
        if( i!=end )
        {
            _visModelBuilder.Update(images[i],depths[i]);
        }
        else
        {
            _visModelBuilder.Update(images[i],depths[i],clouds[i]);
        }
    }    

    size_t n_objs = _visModelBuilder.NumOfModels();
    visualization::PCLVisualizer viewers_cvx[n_objs]; 
    visualization::PCLVisualizer viewers_pts[n_objs]; 
    VisModelBuilder::VisConvexHull models[n_objs];
    for( size_t o=0; o<n_objs; o++ )
    {
        vector<VisModelBuilder::VisConvexHull> chulls(1);
//        _visModelBuilder.GenerateModels(o, chulls);

        VisModelBuilder::ENUM_Criteria flag 
         = VisModelBuilder::C_GRAVITY | VisModelBuilder::C_ONEFACE;
        // = VisModelBuilder::C_GRAVITYONEFACE_TWOFACES;

        _visModelBuilder.GetConvexHull(o, flag, models[o]);

        PolygonMesh polymesh;
        ConvexHull<PointXYZRGB> chull;
        chull.setInputCloud (models[o].cloud_hull);
        chull.reconstruct (*models[o].cloud_hull);
        chull.reconstruct (polymesh);

        stringstream ss_cvx;
        ss_cvx << "convex hull " << o;
        viewers_cvx[o].setWindowName(ss_cvx.str());
        viewers_cvx[o].setSize(600,480);
        viewers_cvx[o].setPosition(600,0);
        viewers_cvx[o].setCameraPosition(0,0,0, 0.7,0,0, 0,0,1);
        viewers_cvx[o].setBackgroundColor (0.2, 0.2, 0.2);
        //viewers_cvx[o].addPolygonMesh(*models[o].polymesh_hull);
        viewers_cvx[o].addPolygonMesh(polymesh);
        viewers_cvx[o].addPointCloud(models[o].cloud);

        stringstream ss_pts;
        ss_pts << "point cloud " << o;
        viewers_pts[o].setWindowName(ss_pts.str());
        viewers_pts[o].setSize(600,480);
        viewers_pts[o].setPosition(600,0);
        viewers_pts[o].setCameraPosition(0,0,0, 0.7,0,0, 0,0,1);
        viewers_pts[o].setBackgroundColor (0.2, 0.2, 0.2);
        viewers_cvx[o].addPointCloud(models[o].cloud_hull, "hull");
        viewers_pts[o].addPointCloud(models[o].cloud,"cloud");
        viewers_pts[o].setPointCloudRenderingProperties(
                       visualization::PCL_VISUALIZER_POINT_SIZE,3,"cloud");
    }
    viewers_cvx[0].spin();
}
