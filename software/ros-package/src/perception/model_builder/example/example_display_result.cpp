#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>

#include <yaml-cpp/yaml.h>

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

#include "utils/utils.hpp"

using namespace std;
using namespace pcl;
using namespace utils;
using namespace Eigen;
namespace po = boost::program_options;

int main(int argc, char* argv[])
{
    int index;
    string prefix;
    string subdir;
    vector<int> range_index;
    int n_objs;

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

    char fp_cloud[256];
    sprintf(fp_cloud,(dp_image + "/" + "%s%06d.cloud.pcd").c_str(),
            prefix.c_str(),index);
    
    PointCloud<PointXYZRGB>::Ptr cloud_input(new PointCloud<PointXYZRGB>);
    if (io::loadPCDFile<PointXYZRGB> (fp_cloud, *cloud_input) == -1)
    {
        ROS_ERROR_STREAM ("Couldn't read file: " << fp_cloud);
        return (-1);
    }
    transformPointCloud(*cloud_input, *cloud_input, tf_cam2world);    

    // Read Result   
    float probs_our2[n_objs][5];

    char fp_res[256];
    sprintf(fp_res,(dp_res + "/%d-%d.result.txt").c_str(), index, index );
    int o2g[n_objs];
    ifstream if_res(fp_res);
    string line;
    for( int l=0; l<6; l++) getline(if_res,line);
    for( int i=0; i<n_objs; i++ )
    {
        int o,g;
        getline(if_res,line);
        sscanf(line.c_str(), "o=%d gt=%d", &o, &g);
        o2g[o] = g;
        getline(if_res,line); // grv
        getline(if_res,line); // two
        getline(if_res,line); // our
        getline(if_res,line); // our2
        char name[256];
        float AoU;
        float AoUs[5];        
        sscanf(line.c_str(), "%s %f %f (%f) %f (%f) %f (%f) %f (%f) %f (%f)",
                            name, &AoU, 
                            &AoUs[0], &probs_our2[o][0],
                            &AoUs[1], &probs_our2[o][1],
                            &AoUs[2], &probs_our2[o][2],
                            &AoUs[3], &probs_our2[o][3],
                            &AoUs[4], &probs_our2[o][4]);
        getline(if_res,line); // our3        
    }
    if_res.close();

    PolygonMesh meshs_grv[n_objs], meshs_two[n_objs], 
                meshs_our[n_objs], meshs_our2[n_objs][5], meshs_our3[n_objs];

    for( int o=0; o<n_objs; o++ )
    {   
        if( o < range_index[0] || range_index[1] < o ) continue;

        sprintf(fp_res,(dp_res + "/%d-%d.grv.%d.result.ply").c_str(), index, index, o);
        string grv(fp_res);        
        if( pcl::io::loadPolygonFilePLY(grv, meshs_grv[o]) == -1 )        
        {
            ROS_ERROR_STREAM ("Couldn't read file: " << fp_res);
            return (-1);
        }
        sprintf(fp_res,(dp_res + "/%d-%d.two.%d.result.ply").c_str(), index, index, o);
        string two(fp_res);        
        if( pcl::io::loadPolygonFilePLY(two, meshs_two[o]) == -1 )        
        {
            ROS_ERROR_STREAM ("Couldn't read file: " << fp_res);
            return (-1);
        }
        sprintf(fp_res,(dp_res + "/%d-%d.our.%d.result.ply").c_str(), index, index, o);
        string our(fp_res);        
        if( pcl::io::loadPolygonFilePLY(our, meshs_our[o]) == -1 )        
        {
            ROS_ERROR_STREAM ("Couldn't read file: " << fp_res);
            return (-1);
        }
        for( int i=0; i<5; i++)
        {
            sprintf(fp_res,(dp_res + "/%d-%d.our2.%d.%d.result.ply").c_str(), index, index, o,i);
            string our2(fp_res);        
            if( pcl::io::loadPolygonFilePLY(our2, meshs_our2[o][i]) == -1 )        
            {
                ROS_ERROR_STREAM ("Couldn't read file: " << fp_res);
                return (-1);
            }
        }
        sprintf(fp_res,(dp_res + "/%d-%d.our3.%d.result.ply").c_str(), index, index, o);
        string our3(fp_res);        
        if( pcl::io::loadPolygonFilePLY(our3, meshs_our3[o]) == -1 )
        {
            ROS_ERROR_STREAM ("Couldn't read file: " << fp_res);
            return (-1);
        }
    }

    // Read ground truth info
    PolygonMesh meshs_gt[n_objs];
    for( int o=0; o<n_objs; o++ )
    {
        if( o < range_index[0] || range_index[1] < o ) continue;

        int g = o2g[o];
        char fp_gt[256];
        sprintf(fp_gt,(dp_res + "/%d-%d.gt.%d.result.ply").c_str(), index, index, g);
                           
        string tmp(fp_gt);
        if( pcl::io::loadPolygonFilePLY(tmp, meshs_gt[o]) == -1 )        
        {
            ROS_ERROR_STREAM ("Couldn't read file: " << fp_gt);
            return (-1);
        }
    }

    visualization::PCLVisualizer viewer; 
    int v1, v2, v3, v4, v5, v6, v7, v8, v9;
    viewer.createViewPort (0.00, 0.66, 0.33, 0.99, v1);
    viewer.createViewPort (0.33, 0.66, 0.66, 0.99, v2);
    viewer.createViewPort (0.66, 0.66, 0.99, 0.99, v3);
    viewer.createViewPort (0.00, 0.33, 0.33, 0.66, v4);
    viewer.createViewPort (0.33, 0.33, 0.66, 0.66, v5);
    viewer.createViewPort (0.66, 0.33, 0.99, 0.66, v6);
    viewer.createViewPort (0.00, 0.00, 0.33, 0.33, v7);
    viewer.createViewPort (0.33, 0.00, 0.66, 0.33, v8);
    viewer.createViewPort (0.66, 0.00, 0.99, 0.33, v9);
    viewer.setWindowName("Display Result");
    viewer.setSize(600,480);
    viewer.setPosition(600,0);
    viewer.setCameraPosition(-1,0,0,1,0,0,0,0,1);
    viewer.setBackgroundColor (1, 1, 1);
    viewer.addPointCloud(cloud_input,"cloud_input", v1);
    viewer.addPointCloud(cloud_input,"cloud_input", v2);
    
    viewer.addPointCloud(cloud_input,"cloud_input", v4);
    viewer.addPointCloud(cloud_input,"cloud_input", v5);
    
    viewer.addPointCloud(cloud_input,"cloud_input", v7);
    viewer.addPointCloud(cloud_input,"cloud_input", v8);
    viewer.addPointCloud(cloud_input,"cloud_input", v9);
    for( int o=0; o<n_objs; o++ )
    {
        if( o < range_index[0] || range_index[1] < o ) continue;
cout << "o=" << o << endl;
        stringstream ss;
        ss << o;
        viewer.addPolygonMesh(meshs_gt[o],  "gt"  + ss.str(), v2);
        viewer.addPolygonMesh(meshs_grv[o], "grv" + ss.str(), v4);
        viewer.addPolygonMesh(meshs_two[o], "two" + ss.str(), v5);
        viewer.addPolygonMesh(meshs_our[o], "our" + ss.str(), v7);
        viewer.addPolygonMesh(meshs_our2[o][0], "our2" + ss.str(), v8);
        viewer.addPolygonMesh(meshs_our3[o], "our3" + ss.str(), v9);

        viewer.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
            colors_vis[o][0]/255.,colors_vis[o][1]/255.,colors_vis[o][2]/255., 
            "gt"+ss.str(), v2);
        viewer.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
            colors_vis[o][0]/255.,colors_vis[o][1]/255.,colors_vis[o][2]/255., 
            "grv"+ss.str(), v4);
        viewer.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
            colors_vis[o][0]/255.,colors_vis[o][1]/255.,colors_vis[o][2]/255., 
            "two"+ss.str(), v5);
        viewer.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
            colors_vis[o][0]/255.,colors_vis[o][1]/255.,colors_vis[o][2]/255., 
            "our"+ss.str(), v7);
        viewer.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
            colors_vis[o][0]/255.,colors_vis[o][1]/255.,colors_vis[o][2]/255., 
            "our2"+ss.str(), v8);
        viewer.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
            colors_vis[o][0]/255.,colors_vis[o][1]/255.,colors_vis[o][2]/255., 
            "our3"+ss.str(), v9);
    }
    

    viewer.spin();

    visualization::PCLVisualizer viewer2; 
    int vs[9];
    viewer2.createViewPort (0.00, 0.66, 0.33, 0.99, vs[0]);
    viewer2.createViewPort (0.33, 0.66, 0.66, 0.99, vs[1]);
    viewer2.createViewPort (0.66, 0.66, 0.99, 0.99, vs[2]);
    viewer2.createViewPort (0.00, 0.33, 0.33, 0.66, vs[3]);
    viewer2.createViewPort (0.33, 0.33, 0.66, 0.66, vs[4]);
    viewer2.createViewPort (0.66, 0.33, 0.99, 0.66, vs[5]);
    viewer2.createViewPort (0.00, 0.00, 0.33, 0.33, vs[6]);
    viewer2.createViewPort (0.33, 0.00, 0.66, 0.33, vs[7]);
    viewer2.createViewPort (0.66, 0.00, 0.99, 0.33, vs[8]);
    viewer2.setWindowName("Display Result");
    viewer2.setSize(600,480);
    viewer2.setPosition(600,0);
    viewer2.setCameraPosition(-1,0,0,1,0,0,0,0,1);
    viewer2.setBackgroundColor (1, 1, 1);    
    for( int o=0; o<n_objs; o++ )
    {
        if( o < range_index[0] || range_index[1] < o ) continue;
cout << "o=" << o << endl;        
        
        for( int i=0; i<5; i++)
        {
            stringstream ss;
            ss << o << i;
            viewer2.addPolygonMesh(meshs_our2[o][0], "our2" + ss.str(), vs[o]);
            viewer2.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
                colors_vis[o][0]/255.,colors_vis[o][1]/255.,colors_vis[o][2]/255., 
                "our2"+ss.str(), vs[o]);        
            viewer2.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_OPACITY , 
                probs_our2[o][i], 
                "our2"+ss.str(), vs[o]);
        }
    }
    

    viewer2.spin();

    return 0;

}
