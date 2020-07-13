#include <iostream>
#include <yaml-cpp/yaml.h>

#include <boost/program_options.hpp>

#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>


#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl_conversions/pcl_conversions.h>

#include "bullet_simulation/bullet_simulation.hpp"
#include "bullet_simulation/conversions.hpp"

#include <btBulletDynamicsCommon.h>
#include "utils/utils.hpp"


using namespace std;
using namespace cv;
using namespace pcl;
using namespace Eigen;
using namespace utils;

namespace po = boost::program_options;

Eigen::Matrix4f tf_cam2world;

int main(int argc, char* argv[])
{
    int index;
    string prefix;
    string subdir;
    string videodir;
    vector<int> range_index;
    vector<int> fixed;
    int n_objs;
    int action;

    po::options_description desc("Example Usage");
    desc.add_options()
        ("help", "help")
        ("index,i",      po::value<int>(&index),
                         "input index")
        ("subdir,s",     po::value<string>(&subdir),
                         "input image director")
        ("prefix,p",     po::value<string>(&prefix)
                         ->default_value(""),
                         "input image filename format")
        ("n_objs,o",      po::value<int>(&n_objs),
                         "# of objects")
        ("indexrange,r", po::value<vector<int> >(&range_index)->multitoken(),
                         "object index range (0,n)")
        ("video,v",     po::value<string>(&videodir),
                         "output video director")
        ("fixed,f",     po::value<vector<int> >(&fixed)->multitoken(),
                         "fixed objects")
        ("action,a",    po::value<int>(&action)->default_value(0),
                         "action" )
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if( vm.count("help") ) 
    {        
        cout << desc << "\n";
        return 0;
    }

    if( range_index.size()==0 )
    {
        range_index.push_back(0); 
        range_index.push_back(n_objs-1);
    }

    if( prefix.compare("")!=0 ) prefix += ".";

    string dp_image("/home/cs1080/projects/3dmodelbuilder/dataset/iros2018/");
    dp_image += (subdir + "/");
    string dp_res("/home/cs1080/projects/3dmodelbuilder/results/iros2018/");
    dp_res += (subdir + "/");
    
    // Read Scene
    Matrix4f tf_cam2world;
    char fp_sceneinfo[256];
    sprintf(fp_sceneinfo, (dp_image + "/%s").c_str(), "scene_info.yaml");

    YAML::Node lconf = YAML::LoadFile(fp_sceneinfo);
    vector<float> camera_RT = lconf["extrinsic"].as<vector<float> >();
    for( int i=0; i<16; i++ ) tf_cam2world(i/4,i%4) = camera_RT[i];        
    
    vector<float> box;
    if( lconf["box"] )
    {
        box = lconf["box"].as<vector<float> >();

        cout << "bbox: ";
        for( int b=0; b<6; b++ )
        {
            cout << box[b] << " ";
        }
        cout << endl;
    } 

    PointCloud<PointXYZRGB>::Ptr cloud_input[n_objs]; 
    PointXYZ centers_input[n_objs];

    PolygonMesh meshs_our[n_objs][32];
    PointCloud<PointXYZRGB>::Ptr meshs_cloud[n_objs][32];
    PointXYZ centers[n_objs][32];
    for( int o=0; o<n_objs; o++ )
    {   
        if( o < range_index[0] || range_index[1] < o ) continue;

        char fp_res[256];
        sprintf(fp_res,(dp_res + "/%s%d-%d.seg.%d.pcd").c_str(), 
                        prefix.c_str(), index, index, o);
        string grv(fp_res);

        cloud_input[o] = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
        if (io::loadPCDFile<PointXYZRGB> (fp_res, *cloud_input[o]) == -1)
        {
            ROS_ERROR_STREAM ("Couldn't read file: " << fp_res);
            return (-1);
        }
        transformPointCloud(*cloud_input[o], *cloud_input[o], tf_cam2world);    
        PointXYZRGB pt_min, pt_max;
        getMinMax3D(*cloud_input[o], pt_min, pt_max);
        double cx = (pt_max.x+pt_min.x)/2, 
               cy = (pt_max.y+pt_min.y)/2, 
               cz = (pt_max.z+pt_min.z)/2;
        for( size_t p=0; p<cloud_input[o]->size(); p++ )
        {
            cloud_input[o]->points[p].x -= cx;
            cloud_input[o]->points[p].y -= cy;
            cloud_input[o]->points[p].z -= cz;
        }        
        centers_input[o].x = cx;
        centers_input[o].y = cy;
        centers_input[o].z = cz;

        for( int i=0; i<32; i++)
        {
            sprintf(fp_res,(dp_res + "/%s%d-%d.our.%d.%d.model.ply").c_str(), 
                            prefix.c_str(), index, index, o,i);
            string our(fp_res); 

            if( pcl::io::loadPolygonFilePLY(our, meshs_our[o][i]) == -1 )        
            {
                ROS_ERROR_STREAM ("Couldn't read file: " << fp_res);
                return (-1);
            }

            PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
            fromPCLPointCloud2(meshs_our[o][i].cloud, *cloud);

            PointXYZRGB pt_min, pt_max;
            getMinMax3D(*cloud, pt_min, pt_max);
            double cx = (pt_max.x+pt_min.x)/2, 
                   cy = (pt_max.y+pt_min.y)/2, 
                   cz = (pt_max.z+pt_min.z)/2;
            for( size_t p=0; p<cloud->size(); p++ )
            {
                cloud->points[p].x -= cx;
                cloud->points[p].y -= cy;
                cloud->points[p].z -= cz;
            }        
            centers[o][i].x = cx;
            centers[o][i].y = cy;
            centers[o][i].z = cz;

            meshs_cloud[o][i] = cloud;
        }
    }
    
    if( fixed.size()==0 )
    {
        for( int o=0; o<n_objs; o++ )
        {
            //int i=0;
            for( int i=0; i<32; i++ )
            {
                float mass = 1;

                MyOpenGLGuiHelper::colors.clear();
                BulletSimulationGui sim;            
                sim.AddBucketShape(box,0);
                sim.AddColor(btVector4(124/255.,97/255.,65/255.,1));
                sim.AddConvexHullShape(*meshs_cloud[o][i], 
                  btVector3(centers[o][i].x,centers[o][i].y,centers[o][i].z), mass);
                sim.AddColor(btVector4( colors_vis[o][0]/255.,
                                        colors_vis[o][1]/255.,
                                        colors_vis[o][2]/255., 1));
                sim.SpinInit();
                sim.ResetCamera( 0.03,-90,60, 0.40,0.025,0 );

                for( int frame=0; frame<20; frame++ )
                {
                    char fp_save[256];
                    sprintf(fp_save,(videodir + "/%s%d-%d.%d.%d.%d.sim.png").c_str(), 
                                     prefix.c_str(), index, index, o,i, frame);       
                    sim.saveNextFrame(fp_save);
                    sim.SpinOnce(0.1);
                }            
                sim.SpinExit();
            }        
        }
    }
    else
    {
        for( int o_chosen=0; o_chosen<n_objs; o_chosen++ )
        {
            {
                bool chosen = false;
                for( int f2=0; f2<fixed.size()/2; f2++ )
                {
                    if( o_chosen==fixed[f2*2] )
                    {
                        chosen = true;
                        break;
                    }
                }
                if( chosen==true ) continue;
            }

            for( int i=0; i<32; i++ )
            {
                MyOpenGLGuiHelper::colors.clear();
                BulletSimulationGui sim;            
                sim.AddBucketShape(box,0);
                sim.AddColor(btVector4(124/255.,97/255.,65/255.,1));

                for( int f=0; f<fixed.size()/2; f++ )
                {
                    int o = fixed[2*f];
                    int i = fixed[2*f+1];

                    sim.AddConvexHullShape(*meshs_cloud[o][i], 
                      btVector3(centers[o][i].x,centers[o][i].y,centers[o][i].z), 0);
                    sim.AddColor(btVector4( colors_vis[o][0]/255.,
                                            colors_vis[o][1]/255.,
                                            colors_vis[o][2]/255., 1));
                    cout << "added: " << o << " " << f << endl;
                }

                for(int o2=0; o2<n_objs; o2++)
                {
                    bool chosen = false;
                    for( int f2=0; f2<fixed.size()/2; f2++ )
                    {
                        if( o_chosen==o2 || o2==fixed[f2*2] )
                        {
                            chosen = true;
                            break;
                        }
                    }
                    if( chosen==false )
                    {
                        sim.AddConvexHullShape(*cloud_input[o2], 
                            btVector3(centers_input[o2].x,centers_input[o2].y,centers_input[o2].z), 0);
                        sim.AddColor(btVector4(0.6,0.6,0.6,1));
                        cout << "added2: " << o2 << endl;

                    }
                }

                sim.AddConvexHullShape(*meshs_cloud[o_chosen][i], 
                    btVector3( centers[o_chosen][i].x,
                               centers[o_chosen][i].y,
                               centers[o_chosen][i].z  ), 1);
                sim.AddColor(btVector4( colors_vis[o_chosen][0]/255.,
                                        colors_vis[o_chosen][1]/255.,
                                        colors_vis[o_chosen][2]/255., 1));

                stringstream ss;
                for( int f=0; f<fixed.size()/2; f++ )
                {
                    ss << fixed[2*f] << "(" << fixed[2*f+1] << ")";
                }

                btRigidBody* body_stick = NULL;
                btVector3 vec_action;
                if( action==1 )
                {
                    btCylinderShape* stick
                     = new btCylinderShapeZ(btVector3(0.005,0.005,0.25));
                    body_stick
                     = sim.CreateRigidBody(stick, btVector3(0.536994,0.17194,-0.20928+0.255),1000);    
                    body_stick->setGravity(btVector3(0,0,0));
                    body_stick->setMassProps(1000,btVector3(0,0,0));
                    sim.AddColor(btVector4(1,1,1,1));
                    vec_action = 4*btVector3(0.482928-0.536994,0.107125-0.17194,0);
                }
                else if( action==2 )
                {
                    btCylinderShape* stick
                     = new btCylinderShapeZ(btVector3(0.005,0.005,0.25));
                    body_stick
                     = sim.CreateRigidBody(stick, btVector3(0.537464,0.0204968,-0.213274+0.255),1000);    
                    body_stick->setGravity(btVector3(0,0,0));
                    body_stick->setMassProps(1000,btVector3(0,0,0));
                    sim.AddColor(btVector4(1,1,1,1));                    
                    vec_action = 4*btVector3(0.493644-0.537464,-0.0521288-0.0204968,0);
                }
                
                sim.SpinInit();
                sim.ResetCamera( 0.03,-90,60, 0.40,0.025,0 );
                for( int frame=0; frame<20; frame++ )
                {
                    char fp_save[256];
                    sprintf(fp_save,(videodir + "/%s%d-%d.%s.%d.%d.%d.sim2.png").c_str(), 
                                     prefix.c_str(), index, index, ss.str().c_str(), o_chosen, i, frame);
                    sim.saveNextFrame(fp_save);

                    if( action > 0) body_stick->setLinearVelocity(vec_action);

                    sim.SpinOnce(0.1);
                }            
                sim.SpinExit();
            }            
        }
    }
}
