#include <cstdlib>
#include <sstream>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

#include "rl_msgs/rl_msgs_visualization.hpp"

using namespace std;
using namespace pcl;

static double colors[][3] = 
{
    {230, 25, 75},
    {60, 180, 75},
    {255, 225,25},
    {0, 130, 200},
    {245, 130, 48},
    {145, 30, 180},
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

static 
void transform(const PointXYZ &pt_in, PointXYZ &pt_out, const Eigen::Matrix4f &tf)
{
    PointXYZ pt;
    pt.x = tf(0,0)*pt_in.x + tf(0,1)*pt_in.y + tf(0,2)*pt_in.z + tf(0,3);
    pt.y = tf(1,0)*pt_in.x + tf(1,1)*pt_in.y + tf(1,2)*pt_in.z + tf(1,3);
    pt.z = tf(2,0)*pt_in.x + tf(2,1)*pt_in.y + tf(2,2)*pt_in.z + tf(2,3);
    pt_out = pt;
}

void AddSegmentationObjects(vector<rl_msgs::SegmentationObject> &segs,
                            visualization::PCLVisualizer &viewer,
                            int viewport, const Eigen::Matrix4f &tf       )
{    
    for( size_t s=0; s<segs.size(); s++ )
    {
        AddSegmentationObject( segs[s], viewer, s, viewport, tf );
    }
}

void AddSegmentationObjects2(vector<rl_msgs::SegmentationObject> &segs,
                             visualization::PCLVisualizer &viewer,
                             int viewport, const Eigen::Matrix4f &tf      )
{    
    for( size_t s=0; s<segs.size(); s++ )
    {
        AddSegmentationObject2( segs[s], viewer, s, viewport, tf );
    }
}

void AddSegmentationObject2( rl_msgs::SegmentationObject &segsv,
                            visualization::PCLVisualizer &viewer, 
                            int idx, int viewport, const Eigen::Matrix4f &tf )
{    
    double r = colors[idx%21][0];
    double g = colors[idx%21][1];
    double b = colors[idx%21][2];

    PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
    fromROSMsg(segsv.cloud, *cloud);
    transformPointCloud(*cloud, *cloud, tf);

    for( size_t p=0; p<cloud->size(); p++ )
    {
        cloud->points[p].r = r;
        cloud->points[p].g = g;
        cloud->points[p].b = b;        
    }

    stringstream ss;
    ss << "[" << idx << "/" << viewport << "]" << "_cloud";
    viewer.addPointCloud(cloud, ss.str(), viewport);
}

void AddSegmentationObject( rl_msgs::SegmentationObject &segsv,
                            visualization::PCLVisualizer &viewer, 
                            int idx, int viewport, const Eigen::Matrix4f &tf )
{    
    double r = colors[idx%21][0]/255.;
    double g = colors[idx%21][1]/255.;
    double b = colors[idx%21][2]/255.;

    PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
    fromROSMsg(segsv.cloud, *cloud);
    transformPointCloud(*cloud, *cloud, tf);

    stringstream ss;
    ss << idx;
    string str_anynumber = ss.str();
    viewer.addPointCloud(cloud, "[" + str_anynumber + "]" + "_cloud",viewport);
    viewer.setPointCloudRenderingProperties(
                 visualization::PCL_VISUALIZER_POINT_SIZE,3,
                                "[" + str_anynumber + "]" + "_cloud",viewport);

    for(size_t f=0; f<segsv.faces.size(); f++)
    {
        stringstream ss;
        ss << "[" << str_anynumber << ",f:" << f+1 << "/" << segsv.faces.size() << "]";
        string name = ss.str();
        
        rl_msgs::SegmentationFace &face = segsv.faces[f];

        PointXYZ pt1_(face.center.x + 
                     face.vecs_axis[0].x * face.sizes[0] + 
                     face.vecs_axis[1].x * face.sizes[1],
                     face.center.y + 
                     face.vecs_axis[0].y * face.sizes[0] + 
                     face.vecs_axis[1].y * face.sizes[1],
                     face.center.z + 
                     face.vecs_axis[0].z * face.sizes[0] + 
                     face.vecs_axis[1].z * face.sizes[1]   );

        PointXYZ pt2_(face.center.x + 
                     face.vecs_axis[0].x * face.sizes[0] - 
                     face.vecs_axis[1].x * face.sizes[1],
                     face.center.y + 
                     face.vecs_axis[0].y * face.sizes[0] - 
                     face.vecs_axis[1].y * face.sizes[1],
                     face.center.z + 
                     face.vecs_axis[0].z * face.sizes[0] - 
                     face.vecs_axis[1].z * face.sizes[1]   );
            
        PointXYZ pt3_(face.center.x - 
                     face.vecs_axis[0].x * face.sizes[0] - 
                     face.vecs_axis[1].x * face.sizes[1],
                     face.center.y - 
                     face.vecs_axis[0].y * face.sizes[0] - 
                     face.vecs_axis[1].y * face.sizes[1],
                     face.center.z - 
                     face.vecs_axis[0].z * face.sizes[0] - 
                     face.vecs_axis[1].z * face.sizes[1]   );

        PointXYZ pt4_(face.center.x - 
                     face.vecs_axis[0].x * face.sizes[0] + 
                     face.vecs_axis[1].x * face.sizes[1],
                     face.center.y - 
                     face.vecs_axis[0].y * face.sizes[0] + 
                     face.vecs_axis[1].y * face.sizes[1],
                     face.center.z - 
                     face.vecs_axis[0].z * face.sizes[0] + 
                     face.vecs_axis[1].z * face.sizes[1]   );

        PointXYZ pt12_(face.center.x + 
                      face.vecs_axis[0].x * face.sizes[0],
                      face.center.y + 
                      face.vecs_axis[0].y * face.sizes[0],
                      face.center.z + 
                      face.vecs_axis[0].z * face.sizes[0]  );

        PointXYZ pt23_(face.center.x + 
                      face.vecs_axis[1].x * face.sizes[1],
                      face.center.y + 
                      face.vecs_axis[1].y * face.sizes[1],
                      face.center.z + 
                      face.vecs_axis[1].z * face.sizes[1]  );

        PointXYZ pt34_(face.center.x - 
                      face.vecs_axis[0].x * face.sizes[0],
                      face.center.y - 
                      face.vecs_axis[0].y * face.sizes[0],
                      face.center.z - 
                      face.vecs_axis[0].z * face.sizes[0]  );

        PointXYZ pt41_(face.center.x - 
                      face.vecs_axis[1].x * face.sizes[1],
                      face.center.y - 
                      face.vecs_axis[1].y * face.sizes[1],
                      face.center.z - 
                      face.vecs_axis[1].z * face.sizes[1]  );

        PointXYZ pt1, pt2, pt3, pt4, pt12, pt23, pt34, pt41;
        transform(pt1_, pt1, tf);
        transform(pt2_, pt2, tf);
        transform(pt3_, pt3, tf);
        transform(pt4_, pt4, tf);
        transform(pt12_, pt12, tf);
        transform(pt23_, pt23, tf);
        transform(pt34_, pt34, tf);
        transform(pt41_, pt41, tf);        

        // box
        viewer.addLine(pt1, pt2, r,g,b, name + "_box12",viewport);
        viewer.addLine(pt2, pt3, r,g,b, name + "_box23",viewport);
        viewer.addLine(pt3, pt4, r,g,b, name + "_box34",viewport);
        viewer.addLine(pt4, pt1, r,g,b, name + "_box41",viewport);
        viewer.addLine(pt12, pt34, 1,0,0, name + "_axis1",viewport);
        viewer.addLine(pt23, pt41, 0,1,0, name + "_axis2",viewport);

        viewer.setShapeRenderingProperties(visualization::PCL_VISUALIZER_LINE_WIDTH, 4, name + "_box12",viewport);
        viewer.setShapeRenderingProperties(visualization::PCL_VISUALIZER_LINE_WIDTH, 4, name + "_box23",viewport);
        viewer.setShapeRenderingProperties(visualization::PCL_VISUALIZER_LINE_WIDTH, 4, name + "_box34",viewport);
        viewer.setShapeRenderingProperties(visualization::PCL_VISUALIZER_LINE_WIDTH, 4, name + "_box41",viewport);

//        viewer.addText3D(name,face.center,face.sizes[1]*0.2, r,g,b, name + "_name",viewport);

        PointXYZ ptc_(face.center.x, face.center.y, face.center.z);
        PointXYZ ptn_(face.center.x + face.normal.x*face.sizes[1]/2., 
                     face.center.y + face.normal.y*face.sizes[1]/2., 
                     face.center.z + face.normal.z*face.sizes[1]/2. );

        PointXYZ ptc, ptn;
        transform(ptc_, ptc, tf);
        transform(ptn_, ptn, tf);

        // normal
        viewer.addLine(ptc, ptn, 0,0,1, name + "_normal",viewport);
    }
}
