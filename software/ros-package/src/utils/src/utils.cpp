#include "utils/utils.hpp"

#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/convex_hull.h>


using namespace std;
using namespace pcl;


namespace utils
{

uint8_t colors_vis[][3] = 
{
    {230, 25, 75}, // [0] red
    {60, 180, 75}, // [1] green
    {255, 225,25}, // [2] yello
    {0, 130, 200}, // [3] blue
    {245, 130, 48},// [4] orange
    {145, 30, 180},// [5] purple
    {70, 240, 240},
    {240, 50, 230},
    {210, 245, 60},
    {250, 190, 190},
    {0, 128, 128},
    {230, 190, 255},
    {170, 110, 40},
    {255, 250, 200},
    {128, 0, 0},
    {170, 255, 195},
    {128, 128, 0},
    {255, 215, 180},
    {0, 0, 128},
    {128, 128, 128},
    {255, 255, 255}
};

float IntersectionOverUnion(const PolygonMesh &polymesh1,
                            const PolygonMesh &polymesh2, 
                            const float resolution )
{
    PointCloud<PointXYZRGB>::Ptr cloud1(new PointCloud<PointXYZRGB>);
    PointCloud<PointXYZRGB>::Ptr cloud2(new PointCloud<PointXYZRGB>);
    PointCloud<PointXYZRGB>::Ptr cloudU(new PointCloud<PointXYZRGB>);
    fromPCLPointCloud2(polymesh1.cloud, *cloud1);
    fromPCLPointCloud2(polymesh2.cloud, *cloud2);        
    *cloudU += *cloud1;
    *cloudU += *cloud2;

    // Generate an uniform grid voxels
    PointXYZRGB pt_min, pt_max;
    getMinMax3D(*cloudU, pt_min, pt_max);    
    int n_x = int((pt_max.x - pt_min.x) / resolution + 1);
    int n_y = int((pt_max.y - pt_min.y) / resolution + 1);
    int n_z = int((pt_max.z - pt_min.z) / resolution + 1);
    typename PointCloud<PointXYZRGB>::Ptr cloud_grid(new PointCloud<PointXYZRGB>);
    cloud_grid->points.resize(n_x*n_y*n_z);
    float z=pt_min.z;
    for(int i_z=0; z<=pt_max.z; i_z++)
    {        
        float y=pt_min.y;
        for(int i_y=0; y<=pt_max.y; i_y++)
        {
            float x=pt_min.x;
            for(int i_x=0; x<=pt_max.x; i_x++)
            {
                int i = n_x*n_y*i_z + n_x*i_y + i_x;                
                PointXYZRGB &pt = cloud_grid->points[i];
                pt.x=x; pt.y=y; pt.z=z;                

                x += resolution;
            }
            y += resolution;
        }
        z += resolution;
    }

    vector<int> idxes1, idxes2, idxesA, idxesU;

    CropHull<PointXYZRGB> cropHull;
    cropHull.setDim(3);
    cropHull.setInputCloud(cloud_grid);

    cropHull.setHullCloud(cloud1);
    cropHull.setHullIndices(polymesh1.polygons);
    cropHull.filter(idxes1);
    
    cropHull.setHullCloud(cloud2);
    cropHull.setHullIndices(polymesh2.polygons);
    cropHull.filter(idxes2);

    sort(idxes1.begin(), idxes1.end());
    sort(idxes2.begin(), idxes2.end());    
    set_intersection( idxes1.begin(),idxes1.end(),
                      idxes2.begin(),idxes2.end(),back_inserter(idxesA));
    set_union(        idxes1.begin(),idxes1.end(),
                      idxes2.begin(),idxes2.end(),back_inserter(idxesU));
        
    const float vol_unit = resolution*resolution*resolution;
    float vol1=idxes1.size()*vol_unit,
          vol2=idxes2.size()*vol_unit,
          volA=idxesA.size()*vol_unit,
          volU=idxesU.size()*vol_unit;

#if 1
    cout << "vol1: " << vol1 << ", "
         << "vol2: " << vol2 << ", "
         << "volA: " << volA << ", "
         << "volU: " << volU << ", "
         << "AoU:  " << (volA/volU) << endl;

    typename PointCloud<PointXYZRGB>::Ptr cloud_tmp1(new PointCloud<PointXYZRGB>);
    typename PointCloud<PointXYZRGB>::Ptr cloud_tmp2(new PointCloud<PointXYZRGB>);
    typename PointCloud<PointXYZRGB>::Ptr cloud_tmpA(new PointCloud<PointXYZRGB>);
    typename PointCloud<PointXYZRGB>::Ptr cloud_tmpU(new PointCloud<PointXYZRGB>);
    for( size_t i=0; i<idxes1.size(); i++ )
    {
        cloud_tmp1->push_back(cloud_grid->points[idxes1[i]]);
    }
    for( size_t i=0; i<idxes2.size(); i++ )
    {
        cloud_tmp2->push_back(cloud_grid->points[idxes2[i]]);
    }
    for( size_t i=0; i<idxesA.size(); i++ )
    {
        cloud_tmpA->push_back(cloud_grid->points[idxesA[i]]);
    }
    for( size_t i=0; i<idxesU.size(); i++ )
    {
        cloud_tmpU->push_back(cloud_grid->points[idxesU[i]]);
    }

    PolygonMesh polymeshA, polymeshU;
    ConvexHull<PointXYZRGB> chull;
    chull.setComputeAreaVolume(true);    
    chull.setInputCloud (cloud_tmpA);
    chull.reconstruct(polymeshA);
    chull.setInputCloud (cloud_tmpU);
    chull.reconstruct(polymeshU);

    visualization::PCLVisualizer viewer; 
    int v1, v2, v3, v4, v5, v6;
    viewer.createViewPort (0.0,  0.0, 0.33, 0.5, v1);
    viewer.createViewPort (0.33, 0.0, 0.66, 0.5, v2);
    viewer.createViewPort (0.66, 0.0, 0.99, 0.5, v3);
    viewer.createViewPort (0.0,  0.5, 0.33, 1.0, v4);
    viewer.createViewPort (0.33, 0.5, 0.66, 1.0, v5);
    viewer.createViewPort (0.66, 0.5, 0.99, 1.0, v6);
    
    viewer.setWindowName("AoverU");
    viewer.setSize(600,480);
    viewer.setPosition(0,0);
    viewer.setCameraPosition(-1,0,0,1,0,0,0,0,1);
    viewer.setBackgroundColor (1,1,1);    
    //viewer.addPointCloud(cloud_tmpA,"cloud_tmpA",v1);
    //viewer.addPointCloud(cloud_tmpU,"cloud_tmpU",v2);
    //viewer.addPointCloud(cloud_tmp1,"cloud_tmp1",v3);
    //viewer.addPointCloud(cloud_tmp2,"cloud_tmp2",v4);

    viewer.addPolygonMesh(polymeshA,"polymeshA",v1);
    viewer.addPolygonMesh(polymeshU,"polymeshU",v2);
    viewer.addPolygonMesh(polymesh1,"polymesh1",v3);
    viewer.addPolygonMesh(polymesh2,"polymesh2",v4);
    viewer.addPolygonMesh(polymesh1,"polymesh11",v5);
    viewer.addPolygonMesh(polymesh2,"polymesh22",v5);
    
    viewer.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
        colors_vis[5][0]/255.,colors_vis[5][1]/255.,colors_vis[5][2]/255., 
        "polymeshA", v1);
    viewer.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
        colors_vis[2][0]/255.,colors_vis[2][1]/255.,colors_vis[2][2]/255., 
        "polymeshU", v2);
    viewer.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
        colors_vis[0][0]/255.,colors_vis[0][1]/255.,colors_vis[0][2]/255., 
        "polymesh1", v3);
    viewer.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
        colors_vis[3][0]/255.,colors_vis[3][1]/255.,colors_vis[3][2]/255., 
        "polymesh2", v4);
    viewer.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
        colors_vis[0][0]/255.,colors_vis[0][1]/255.,colors_vis[0][2]/255., 
        "polymesh11", v5);
    viewer.setPointCloudRenderingProperties( visualization::PCL_VISUALIZER_COLOR, 
        colors_vis[3][0]/255.,colors_vis[3][1]/255.,colors_vis[3][2]/255., 
        "polymesh22", v5);


/*
    viewer.addText3D("A", cloud_tmpA->points[0], 0.01, 0,1,0, "name_A", v1);
    viewer.addText3D("U", cloud_tmpU->points[0], 0.01, 1,0,1, "name_U", v2);
    viewer.addText3D("1", cloud_tmp1->points[0], 0.01, 1,0,0, "name_1", v3);
    viewer.addText3D("2", cloud_tmp2->points[0], 0.01, 0,0,1, "name_2", v4);
*/
    viewer.spin();
#endif

    return volA / volU;
}

}