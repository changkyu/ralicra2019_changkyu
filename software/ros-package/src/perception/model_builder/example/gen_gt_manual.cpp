#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>

#include <yaml-cpp/yaml.h>

#include <cv_bridge/cv_bridge.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/PolygonMesh.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>

#include "bullet_simulation/bullet_simulation.hpp"
#include "bullet_simulation/conversions.hpp"

#include "model_builder/vis_model_builder.hpp"

#include <shape_msgs/Plane.h>
#include <btBulletDynamicsCommon.h>

using namespace std;
using namespace cv;
using namespace pcl;
using namespace Eigen;

namespace po = boost::program_options;

Matrix4f tf_cam2world;

struct GT
{
    string name;
    Quaternionf q;
    Vector3f trans;
    Matrix4f mat;
} GT;

void toTransformationMatrix(Quaternionf &q, Vector3f trans, Matrix4f& mat)
{
    mat(0,3) = trans(0);
    mat(1,3) = trans(1);
    mat(2,3) = trans(2);        
    mat(3,0) = 0; mat(3,1) = 0; mat(3,2) = 0; mat(3,3) = 1;

    Matrix3f rotMat;
    rotMat = q.toRotationMatrix();

    for(int ii = 0;ii < 3; ii++)
        for(int jj=0; jj < 3; jj++)
            mat(ii,jj) = rotMat(ii,jj);
}

void toGT(Matrix4f &mat, struct GT &gt)
{
    Quaternionf q(mat.block<3,3>(0,0));
    gt.q = q.normalized();
    gt.trans = mat.block<3,1>(0,3);
    toTransformationMatrix(gt.q, gt.trans, gt.mat);
}

int idx_gt=0;
vector<struct GT> gts;
vector<PointCloud<PointXYZRGB>::Ptr> meshs_gt;
vector<PointCloud<PointXYZRGB>::Ptr> meshs_gt_tran;
visualization::PCLVisualizer viewer;
int v1, v2;
void Update()
{
    for( size_t g=0; g<gts.size(); g++ )
    {        
        transformPointCloud(*meshs_gt[g], *meshs_gt_tran[g], gts[g].mat);

        stringstream ss;
        ss << "gt" << g;
        viewer.removePointCloud(ss.str(), v1);        
        viewer.removePointCloud(ss.str() + "v2", v2);        
        viewer.addPointCloud(meshs_gt_tran[g], ss.str(), v1);
        viewer.addPointCloud(meshs_gt_tran[g], ss.str() + "v2", v2);
    }    
}

void Callback_pclkeyboard (const pcl::visualization::KeyboardEvent &event,
                           void* viewer_void)
{
    if (event.keyDown())
    {
        if( event.getKeySym()=="space" )
        {
            idx_gt = (idx_gt + 1) % gts.size();
            cout << gts[idx_gt].name << endl;
        }
        else if( event.getKeySym()=="Return" )
        {
            cout << "------------------------------" << endl;

            for( size_t g=0; g<gts.size(); g++ )
            {
                cout << gts[g].name << " ";
                cout << gts[g].trans(0) << " ";
                cout << gts[g].trans(1) << " ";
                cout << gts[g].trans(2) << " ";
                cout << gts[g].q.w() << " ";                
                cout << gts[g].q.x() << " ";
                cout << gts[g].q.y() << " ";
                cout << gts[g].q.z() << endl;
            }

            cout << "------------------------------" << endl;
        }
        else
        {   
            char keyCode = event.getKeyCode();
            float angle_x=0, angle_y=0, angle_z=0;
            float trans_x=0, trans_y=0, trans_z=0;

            if( keyCode=='w' )
            {
                angle_y =  M_PI/180*1;
            }
            else if( keyCode=='s' )
            {
                angle_y = -M_PI/180*1;
            }
            else if( keyCode=='d' )
            {
                angle_x =  M_PI/180*1;
            }
            else if( keyCode=='a' )
            {
                angle_x = -M_PI/180*1;
            }
            else if( keyCode=='0' )
            {
                angle_z =  M_PI/180*1;
            }
            else if( keyCode=='.' )
            {
                angle_z = -M_PI/180*1;
            }
            else if( keyCode=='8' )
            {
                trans_x =  0.001;
            }
            else if( keyCode=='2' )
            {
                trans_x = -0.001;
            }
            else if( keyCode=='4' )
            {
                trans_y =  0.001;
            }
            else if( keyCode=='6' )
            {
                trans_y = -0.001;
            }
            else if( keyCode=='9' )
            {
                trans_z =  0.001;
            }
            else if( keyCode=='3' )
            {
                trans_z = -0.001;
            }

            if( event.isAltPressed() )
            {
                angle_x *= 10; angle_y *= 10; angle_z *= 10;
                trans_x *= 10; trans_y *= 10; trans_z *= 10;
            }

            Matrix3f m;
            m = AngleAxisf(angle_x, Vector3f::UnitX())
              * AngleAxisf(angle_y, Vector3f::UnitY())
              * AngleAxisf(angle_z, Vector3f::UnitZ());

            Matrix4f m2org;
            m2org << 1, 0, 0, -gts[idx_gt].trans(0),
                     0, 1, 0, -gts[idx_gt].trans(1),
                     0, 0, 1, -gts[idx_gt].trans(2),
                     0, 0, 0, 1;
            Matrix4f m2back;
            m2back << 1, 0, 0, gts[idx_gt].trans(0),
                      0, 1, 0, gts[idx_gt].trans(1),
                      0, 0, 1, gts[idx_gt].trans(2),
                      0, 0, 0, 1;

            Matrix4f mat_new;
            mat_new.block<3,3>(0,0) = m;
            mat_new(0,3) = trans_x;
            mat_new(1,3) = trans_y;
            mat_new(2,3) = trans_z;
            mat_new(3,0)=0; mat_new(3,1)=0; mat_new(3,2)=0; mat_new(3,3)=1;

            Matrix4f tf = m2back * mat_new * m2org * gts[idx_gt].mat;
            toGT(tf, gts[idx_gt]);
        }       

        Update();
    }
}

int main(int argc, char* argv[])
{    
    string dp_image;    
    string prefix;
    int idx;
    
    po::options_description desc("Manual Alignment tool");
    desc.add_options()
        ("help", "help")
        ("inputdir,d",   po::value<string>(&dp_image),
                         "input image director")        
        ("prefix,p",     po::value<string>(&prefix)
                         ->default_value(""),
                         "input image filename format")
        ("index,i",      po::value<int>(&idx),
                         "input image index")        
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if( vm.count("help")        ||
        dp_image.compare("")==0    ) 
    {
        cout << dp_image << endl;
        cout << prefix << endl;
        cout << idx << endl;
        cout << desc << "\n";
        return 0;
    }

    if( prefix.compare("") ) prefix += ".";
    
    // Read iargcts    
    PointCloud<PointXYZRGB>::Ptr cloud_cam(new PointCloud<PointXYZRGB>);
    
    char fp_cloud[256];
    sprintf(fp_cloud,(dp_image + "/" + "%s%06d.cloud.pcd").c_str(),
            prefix.c_str(),idx);        
    if (io::loadPCDFile<PointXYZRGB> (fp_cloud, *cloud_cam) == -1)
    {
        ROS_ERROR_STREAM ("Couldn't read file: " << fp_cloud);
        return (-1);
    }
    cout << fp_cloud << endl;

    char fp_sceneinfo[256];
    sprintf(fp_sceneinfo, (dp_image + "/%s").c_str(), "scene_info.yaml");
    YAML::Node lconf = YAML::LoadFile(fp_sceneinfo);    
    std::vector<float> camera_RT = lconf["extrinsic"].as<vector<float> >();            
    
    for( int i=0; i<16; i++ ) tf_cam2world(i/4,i%4) = camera_RT[i];
    cout << tf_cam2world << endl;

    char fp_gt[256];
    sprintf(fp_gt,(dp_image + "/" + "%s%06d.gt.txt").c_str(),
            prefix.c_str(),idx);
    cout << fp_gt << endl;
    ifstream if_gt(fp_gt);    
    string line;
    while( getline(if_gt,line) )
    {
        istringstream iss(line);

        struct GT gt;        
        iss >> gt.name;
        iss >> gt.trans(0);
        iss >> gt.trans(1);
        iss >> gt.trans(2);
        iss >> gt.q.w();
        iss >> gt.q.x();
        iss >> gt.q.y();
        iss >> gt.q.z();
        toTransformationMatrix(gt.q, gt.trans, gt.mat);
    
        cout << gt.name << ": ";
        cout << gt.trans(0) << " ";
        cout << gt.trans(1) << " ";
        cout << gt.trans(2) << " ";
        cout << gt.q.w() << " ";                
        cout << gt.q.x() << " ";
        cout << gt.q.y() << " ";
        cout << gt.q.z() << endl;
        cout << gt.mat << endl;        

        gts.push_back(gt);
    }
    if_gt.close();

    if( gts.size()==0 )
    {
        cout << "No such a gt file: " << fp_gt << endl;
    }

    meshs_gt.resize(gts.size());
    meshs_gt_tran.resize(gts.size());
    for( size_t g=0; g<gts.size(); g++ )
    {
        meshs_gt[g] = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
        meshs_gt_tran[g] = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
        sprintf(fp_gt,"/home/cs1080/projects/dataset/YCB/ycb/%s/clouds/merged_cloud.ply",
                       gts[g].name.c_str());        
        string tmp(fp_gt);        
        if( pcl::io::loadPLYFile(tmp, *meshs_gt[g]) == -1 )        
        {
            ROS_ERROR_STREAM ("Couldn't read file: " << fp_gt);
            return (-1);        
        }

        VoxelGrid<PointXYZRGB> sor;
        sor.setInputCloud (meshs_gt[g]);
        sor.setLeafSize (0.003, 0.003, 0.003);
        sor.filter (*meshs_gt[g]);

        cout << fp_gt << endl;
    }
       
    cout << gts[idx_gt].name << endl;
    viewer.createViewPort (0.00, 0.00, 0.50, 1.00, v1);
    viewer.createViewPort (0.50, 0.00, 1.00, 1.00, v2);
    viewer.setWindowName("ground truth");
    viewer.setSize(600,480);
    viewer.setPosition(600,0);
    viewer.setCameraPosition(-1,0,0,1,0,0,0,0,1);
    viewer.setBackgroundColor (0.2, 0.2, 0.2);
    viewer.registerKeyboardCallback(Callback_pclkeyboard);
    PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
    transformPointCloud(*cloud_cam, *cloud, tf_cam2world);
    viewer.addPointCloud(cloud, "cloud", v1);    
    for( size_t g=0; g<gts.size(); g++ )
    {        
        transformPointCloud(*meshs_gt[g], *meshs_gt_tran[g], gts[g].mat);
        
        stringstream ss;
        ss << "gt" << g;
        viewer.addPointCloud(meshs_gt_tran[g], ss.str(), v1);
        viewer.addPointCloud(meshs_gt_tran[g], ss.str() + "v2", v2);
    }
    viewer.spin();

    cout << "------------------------------" << endl;

    for( size_t g=0; g<gts.size(); g++ )
    {
        cout << gts[g].name << " ";
        cout << gts[g].trans(0) << " ";
        cout << gts[g].trans(1) << " ";
        cout << gts[g].trans(2) << " ";
        cout << gts[g].q.w() << " ";                
        cout << gts[g].q.x() << " ";
        cout << gts[g].q.y() << " ";
        cout << gts[g].q.z() << endl;
    }

    cout << "------------------------------" << endl;

    return 0;

}