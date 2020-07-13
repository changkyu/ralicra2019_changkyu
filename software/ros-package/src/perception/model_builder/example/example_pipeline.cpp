#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>

#include <yaml-cpp/yaml.h>

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
    string videodir;
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
        ("video,v",     po::value<string>(&videodir),
                         "output video director")
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

    PointCloud<PointXYZRGB>::Ptr cloud_all(new PointCloud<PointXYZRGB>);
    PointCloud<PointXYZRGB>::Ptr cloud_seg(new PointCloud<PointXYZRGB>);
    PointCloud<PointXYZRGB>::Ptr cloud_input[n_objs]; 
    PolygonMesh meshs_our[n_objs][32];
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

        *cloud_all += *cloud_input[o];

        PointCloud<PointXYZRGB> cloud_seg_tmp;
        copyPointCloud(*cloud_input[o], cloud_seg_tmp);
        for( int p=0; p<cloud_seg_tmp.size(); p++ )
        {
            cloud_seg_tmp[p].r = colors_vis[o][0];
            cloud_seg_tmp[p].g = colors_vis[o][1];
            cloud_seg_tmp[p].b = colors_vis[o][2];
        }
        *cloud_seg += cloud_seg_tmp;

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
            centers[o][i].x = cx;
            centers[o][i].y = cy;
            centers[o][i].z = cz;
        }
    }

    PointXYZ center;
    PointXYZRGB pt_min, pt_max;
    getMinMax3D(*cloud_all, pt_min, pt_max);
    double cx = (pt_max.x+pt_min.x)/2, 
           cy = (pt_max.y+pt_min.y)/2, 
           cz = (pt_max.z+pt_min.z)/2;
    center.x = cx;
    center.y = cy;
    center.z = cz;

#if 1
    int v1, v2;    
    visualization::PCLVisualizer viewer;     
    viewer.createViewPort (0.00, 0.00, 0.50, 1.00, v1);
    viewer.createViewPort (0.50, 0.00, 1.00, 1.00, v2);
    viewer.setSize(600,480);
    viewer.setPosition(600,0);
    viewer.setCameraPosition(-1,0,0,1,0,0,0,0,1);
    viewer.setBackgroundColor (1, 1, 1);
    viewer.addPointCloud(cloud_all, "cloud_all", v1);
    viewer.addPointCloud(cloud_seg, "cloud_seg", v2);
    viewer.setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE,
         2, "cloud_all", v1);
    viewer.setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE,
         2, "cloud_seg", v2);

    int frame=0;
    float a=0;
    while( a <= 4*M_PI )
    {
        float x = -cos(a);
        float y = sin(a);
        float z = 0.50;

        viewer.setCameraPosition(center.x+x*1.5,
                                 center.y+y*1.5,
                                 center.z+z*1.5,
                                 center.x,
                                 center.y,
                                 center.z,
                                 0,0,1, v1);

        a = a + 0.05; 
        viewer.spinOnce(1);
        
        char fp_save[256];
        sprintf(fp_save,(videodir + "/%s%d-%d.%d.seg.png").c_str(), 
                         prefix.c_str(), index, index, frame);            
        viewer.saveScreenshot(fp_save);
        frame++;
    }    

    visualization::PCLVisualizer viewer2;     
    viewer2.createViewPort (0.00, 0.00, 0.50, 1.00, v1);
    viewer2.createViewPort (0.50, 0.00, 1.00, 1.00, v2);
    viewer2.setSize(600,480);
    viewer2.setPosition(600,0);
    viewer2.setCameraPosition(-1,0,0,1,0,0,0,0,1);
    viewer2.setBackgroundColor (1, 1, 1);
    
    for( int o=0; o<n_objs; o++ )
    {
        viewer2.removeAllPointClouds(v1);        
        viewer2.addPointCloud(cloud_input[o], "cloud_input1", v1);        
        viewer2.setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE,
         2, "cloud_input1", v1);

        int frame=0;
        float a=0;
        for( int i=0; i<32; i++ )
        {
            viewer2.removeAllPointClouds(v2);
            viewer2.addPointCloud(cloud_input[o], "cloud_input2", v2);
            viewer2.addPolygonMesh(meshs_our[o][i], "polymesh", v2);
            viewer2.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
                colors_vis[o][0]/255.,colors_vis[o][1]/255.,colors_vis[o][2]/255., 
                "polymesh", v2);

            for( int tmp=0; tmp<5; tmp++)
            {
                float x = -cos(a);
                float y = sin(a);
                float z = 0.50;

                viewer2.setCameraPosition(centers[o][0].x+x*1.5,
                                          centers[o][0].y+y*1.5,
                                          centers[o][0].z+z*1.5,
                                          centers[o][0].x,
                                          centers[o][0].y,
                                          centers[o][0].z,
                                          0,0,1, v1);

                a = a + 0.05; 
                viewer2.spinOnce(1);
                
                char fp_save[256];
                sprintf(fp_save,(videodir + "/%s%d-%d.%d.%d.hyp.png").c_str(), 
                                 prefix.c_str(), index, index, o, frame);
                viewer2.saveScreenshot(fp_save);
                frame++;
            }
        }        
    }    

    visualization::PCLVisualizer viewer3;     
    viewer3.createViewPort (0.00, 0.00, 0.50, 1.00, v1);
    viewer3.createViewPort (0.50, 0.00, 1.00, 1.00, v2);
    viewer3.setSize(600,480);
    viewer3.setPosition(600,0);
    viewer3.setCameraPosition(-1,0,0,1,0,0,0,0,1);
    viewer3.setBackgroundColor (1, 1, 1);
    viewer3.addPointCloud(cloud_all, "cloud_all", v1);
    viewer3.addPointCloud(cloud_all, "cloud_all2", v2);
    
    viewer3.addPolygonMesh(meshs_our[1][1],  "polymesh1", v2);
    viewer3.addPolygonMesh(meshs_our[6][20],  "polymesh6", v2);
    viewer3.addPolygonMesh(meshs_our[5][11], "polymesh5", v2);
    viewer3.addPolygonMesh(meshs_our[3][4],  "polymesh3", v2);
    viewer3.addPolygonMesh(meshs_our[2][9],  "polymesh2", v2);
    viewer3.addPolygonMesh(meshs_our[4][7],  "polymesh4", v2);
    viewer3.addPolygonMesh(meshs_our[0][8],  "polymesh0", v2);

    viewer3.setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE,
         2, "cloud_all", v1);
    viewer3.setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE,
         2, "cloud_all2", v2);
    for( int o=0; o<n_objs; o++ )
    {
        stringstream ss;
        ss << "polymesh" << o;
        viewer3.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
                colors_vis[o][0]/255.,colors_vis[o][1]/255.,colors_vis[o][2]/255., 
                ss.str(), v2);
    }

    frame=0;
    a=0;
    while( a <= 4*M_PI )
    {
        float x = -cos(a);
        float y = sin(a);
        float z = 0.50;

        viewer3.setCameraPosition(center.x+x*1.5,
                                 center.y+y*1.5,
                                 center.z+z*1.5,
                                 center.x,
                                 center.y,
                                 center.z,
                                 0,0,1, v1);

        a = a + 0.05; 
        viewer3.spinOnce(1);
        
        char fp_save[256];
        sprintf(fp_save,(videodir + "/%s%d-%d.%d.res.png").c_str(), 
                         prefix.c_str(), index, index, frame);            
        viewer3.saveScreenshot(fp_save);
        frame++;
    }    

#else
    visualization::PCLVisualizer viewer; 
    //int v1, v2, v3, v4, v5, v6, v7, v8, v9;
    int v1, v2, v3;
    int vs[32];
#if 0    
    viewer.createViewPort (0.00, 0.66, 0.33, 0.99, v1);
    viewer.createViewPort (0.33, 0.66, 0.66, 0.99, v2);
    viewer.createViewPort (0.66, 0.66, 0.99, 0.99, v3);
    viewer.createViewPort (0.00, 0.33, 0.33, 0.66, v4);
    viewer.createViewPort (0.33, 0.33, 0.66, 0.66, v5);
    viewer.createViewPort (0.66, 0.33, 0.99, 0.66, v6);
    viewer.createViewPort (0.00, 0.00, 0.33, 0.33, v7);
    viewer.createViewPort (0.33, 0.00, 0.66, 0.33, v8);
    viewer.createViewPort (0.66, 0.00, 0.99, 0.33, v9);
#else
    viewer.createViewPort (0.00, 0.66, 0.33, 0.99, v1);
    viewer.createViewPort (0.33, 0.66, 0.66, 0.99, v2);
    viewer.createViewPort (0.66, 0.66, 0.99, 0.99, v3);

    viewer.createViewPort (0.000, 0.495, 0.125, 0.660, vs[0]);
    viewer.createViewPort (0.125, 0.495, 0.250, 0.660, vs[1]);
    viewer.createViewPort (0.250, 0.495, 0.375, 0.660, vs[2]);
    viewer.createViewPort (0.375, 0.495, 0.500, 0.660, vs[3]);
    viewer.createViewPort (0.500, 0.495, 0.625, 0.660, vs[4]);
    viewer.createViewPort (0.625, 0.495, 0.750, 0.660, vs[5]);
    viewer.createViewPort (0.750, 0.495, 0.875, 0.660, vs[6]);
    viewer.createViewPort (0.875, 0.495, 1.000, 0.660, vs[7]);
    
    viewer.createViewPort (0.000, 0.330, 0.125, 0.495, vs[8]);
    viewer.createViewPort (0.125, 0.330, 0.250, 0.495, vs[9]);
    viewer.createViewPort (0.250, 0.330, 0.375, 0.495, vs[10]);
    viewer.createViewPort (0.375, 0.330, 0.500, 0.495, vs[11]);
    viewer.createViewPort (0.500, 0.330, 0.625, 0.495, vs[12]);
    viewer.createViewPort (0.625, 0.330, 0.750, 0.495, vs[13]);
    viewer.createViewPort (0.750, 0.330, 0.875, 0.495, vs[14]);
    viewer.createViewPort (0.875, 0.330, 1.000, 0.495, vs[15]);

    viewer.createViewPort (0.000, 0.165, 0.125, 0.330, vs[16]);
    viewer.createViewPort (0.125, 0.165, 0.250, 0.330, vs[17]);
    viewer.createViewPort (0.250, 0.165, 0.375, 0.330, vs[18]);
    viewer.createViewPort (0.375, 0.165, 0.500, 0.330, vs[19]);
    viewer.createViewPort (0.500, 0.165, 0.625, 0.330, vs[20]);
    viewer.createViewPort (0.625, 0.165, 0.750, 0.330, vs[21]);
    viewer.createViewPort (0.750, 0.165, 0.875, 0.330, vs[22]);
    viewer.createViewPort (0.875, 0.165, 1.000, 0.330, vs[23]);

    viewer.createViewPort (0.000, 0.000, 0.125, 0.165, vs[24]);
    viewer.createViewPort (0.125, 0.000, 0.250, 0.165, vs[25]);
    viewer.createViewPort (0.250, 0.000, 0.375, 0.165, vs[26]);
    viewer.createViewPort (0.375, 0.000, 0.500, 0.165, vs[27]);
    viewer.createViewPort (0.500, 0.000, 0.625, 0.165, vs[28]);
    viewer.createViewPort (0.625, 0.000, 0.750, 0.165, vs[29]);
    viewer.createViewPort (0.750, 0.000, 0.875, 0.165, vs[30]);
    viewer.createViewPort (0.875, 0.000, 1.000, 0.165, vs[31]);

#endif
    viewer.setWindowName("Display Result");
    viewer.setSize(600,480);
    viewer.setPosition(600,0);
    viewer.setCameraPosition(-1,0,0,1,0,0,0,0,1);
    viewer.setBackgroundColor (1, 1, 1);

    for( int o=0; o<n_objs; o++ )
    {
        if( o < range_index[0] || range_index[1] < o ) continue;

        cout << "o=" << o << endl;

        viewer.removeAllPointClouds(v1);
        viewer.removeAllPointClouds(v2);
        viewer.removeAllPointClouds(v3);

        stringstream ss;
        ss << o;

        viewer.addPointCloud(cloud_all, "cloud_all" + ss.str(), v1);
        viewer.addPointCloud(cloud_seg, "cloud_seg" + ss.str(), v2);
        viewer.addPointCloud(cloud_input[o], "cloud_cur" + ss.str(), v3);
        for( int i=0; i<32; i++ )
        {
            stringstream ss2;
            ss2 << "o=" << o << "i=" << i;
            string id = ss2.str();
            
            viewer.removeAllPointClouds(vs[i]);
            viewer.addPolygonMesh(meshs_our[o][i], id, vs[i]);
            viewer.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
                colors_vis[o][0]/255.,colors_vis[o][1]/255.,colors_vis[o][2]/255., 
                id, vs[i]);
        }
            
        int frame=0;
        float a=0;
        while( a <= 4*M_PI )
        {
            float x = -cos(a);
            float y = sin(a);
            float z = 0.50;
#if 1
            viewer.setCameraPosition(center.x+x*0.7,
                                     center.y+y*0.7,
                                     center.z+z*0.7,
                                     center.x,
                                     center.y,
                                     center.z,
                                     0,0,1, v1);
#else
            viewer.setCameraPosition(center.x+x*2,
                                     center.y+y*2,
                                     center.z+z*2,
                                     center.x,
                                     center.y,
                                     center.z,
                                     0,0,1, v1);
            viewer.setCameraPosition(center.x+x*2,
                                     center.y+y*2,
                                     center.z+z*2,
                                     center.x,
                                     center.y,
                                     center.z,
                                     0,0,1, v2);
            viewer.setCameraPosition(center.x+x*2,
                                     center.y+y*2,
                                     center.z+z*2,
                                     center.x,
                                     center.y,
                                     center.z,
                                     0,0,1, v3);

            for( int i=0; i<32; i++ )
            {
                viewer.setCameraPosition(centers[o][i].x+x*0.5,
                                         centers[o][i].y+y*0.5,
                                         centers[o][i].z+z*0.5,
                                         centers[o][i].x,
                                         centers[o][i].y,
                                         centers[o][i].z,
                                         0,0,1, vs[i]);
            }
#endif
            a = a + 0.1; 
            viewer.spinOnce(1);
            
            char fp_save[256];
            sprintf(fp_save,(videodir + "/%s%d-%d.%d.%d.hyp.png").c_str(), 
                             prefix.c_str(), index, index, o, frame);            
            viewer.saveScreenshot(fp_save);
            frame++;
        }
        
        for( int j=0; j<8; j++ )
        {   
            viewer.removeAllPointClouds(vs[j]);
        }
    }    
#endif
    return 0;

}
