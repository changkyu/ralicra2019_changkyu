// PCL
#include <pcl/segmentation/lccp_segmentation.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// VTK
#include <vtkSmartPointer.h>
#include <vtkCellArray.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyLine.h>

#include "segmentation/vis.hpp"

using namespace std;
using namespace pcl;
using namespace cv;

typedef PointXYZRGB PointT;
typedef PointCloud<PointT> PointCloudT;

static double colors_vis[][3] = 
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

void addSupervoxelToViewer(  
  map<uint32_t, Supervoxel<PointT>::Ptr> &supervoxel_clusters,
  pcl::visualization::PCLVisualizer &viewer                     )
{
    viewer.removeAllPointClouds();

    for(map<uint32_t,Supervoxel<PointT>::Ptr>::iterator
          it_map = supervoxel_clusters.begin();
        it_map != supervoxel_clusters.end(); 
        it_map++                                          )
    {   
        PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
        
        for(PointCloud<PointT>::iterator it_pt=it_map->second->voxels_->begin(); 
            it_pt!=it_map->second->voxels_->end(); it_pt++)
        {
            PointXYZRGB pt;
            pt.x = it_pt->x;
            pt.y = it_pt->y;
            pt.z = it_pt->z;
            //pt.label = it_map->first;
            pt.r = colors_vis[it_map->first%21][0];
            pt.g = colors_vis[it_map->first%21][1];
            pt.b = colors_vis[it_map->first%21][2];
            cloud->push_back(pt);
        }

        stringstream ss;
        ss << it_map->first;
        viewer.addPointCloud(cloud, ss.str() );
        viewer.setPointCloudRenderingProperties(
                 visualization::PCL_VISUALIZER_POINT_SIZE,3,ss.str());
    }
}

void
addSupervoxelConnectionToViewer (PointT &supervoxel_center, 
                                 PointT &adjacent_supervoxel_center,
                                 float weight,
                                 std::string supervoxel_name,
                                 pcl::visualization::PCLVisualizer &viewer)
{
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New (); 
  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New (); 
  vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New ();
  
  points->InsertNextPoint (supervoxel_center.data);
  points->InsertNextPoint (adjacent_supervoxel_center.data);
  
  // Create a polydata to store everything in
  vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New ();
  // Add the points to the dataset
  polyData->SetPoints (points);
  polyLine->GetPointIds()->SetNumberOfIds(points->GetNumberOfPoints ());
  for(unsigned int i = 0; i < points->GetNumberOfPoints (); i++)
    polyLine->GetPointIds ()->SetId (i,i);
  cells->InsertNextCell (polyLine);
  // Add the lines to the dataset
  polyData->SetLines (cells);
  viewer.addModelFromPolyData (polyData,supervoxel_name);
  viewer.setShapeRenderingProperties(visualization::PCL_VISUALIZER_LINE_WIDTH,
                                     2*weight,supervoxel_name);
  viewer.setShapeRenderingProperties(visualization::PCL_VISUALIZER_COLOR,
                                     0,0,1,supervoxel_name);
}

void addSupervoxelGraphToViewer(
    std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > &supervoxel_clusters,
    std::multimap<uint32_t,uint32_t> &label_adjacency,
    map<pair<uint32_t,uint32_t>, float> &weight_adjacency,
    pcl::visualization::PCLVisualizer &viewer
)
{
    std::multimap<uint32_t,uint32_t>::iterator label_itr=label_adjacency.begin();
    for ( ; label_itr != label_adjacency.end() ; )
    {
        //First get the label 
        uint32_t supervoxel_label = label_itr->first;
        if(supervoxel_clusters.find(supervoxel_label)==supervoxel_clusters.end())
        {
            cout << supervoxel_label << " not found" << endl;
            label_itr = label_adjacency.upper_bound (supervoxel_label);
            continue;
        }

         //Now get the supervoxel corresponding to the label
        pcl::Supervoxel<PointT>::Ptr supervoxel
         = supervoxel_clusters.at(supervoxel_label);

        //Now we make a name for this polygon
        PointT pt_ctr;
        pt_ctr.x=supervoxel->centroid_.x;
        pt_ctr.y=supervoxel->centroid_.y;
        pt_ctr.z=supervoxel->centroid_.z;

        // normal
        PointT pt_nml;
        pt_nml.x = 0.02*supervoxel->normal_.normal_x + pt_ctr.x;
        pt_nml.y = 0.02*supervoxel->normal_.normal_y + pt_ctr.y;
        pt_nml.z = 0.02*supervoxel->normal_.normal_z + pt_ctr.z;
        stringstream ss_line;
        ss_line << "line_" << supervoxel_label;
//        viewer.addLine(pt_ctr,pt_nml,0,1,1, ss_line.str());
        
        //Now we need to iterate through the adjacent supervoxels and make a point cloud of them
        PointCloudT adjacent_supervoxel_centers;
        pair< multimap<uint32_t,uint32_t>::iterator,
              multimap<uint32_t,uint32_t>::iterator  > ret
         = label_adjacency.equal_range(supervoxel_label);
        for (multimap<uint32_t,uint32_t>::iterator adjacent_itr = ret.first;
             adjacent_itr!=ret.second; adjacent_itr++)
        {     
          if( label_itr->first > adjacent_itr->second ||
            supervoxel_clusters.find(adjacent_itr->second)==supervoxel_clusters.end())
          {
            continue;
          }

          pcl::Supervoxel<PointT>::Ptr neighbor_supervoxel
           = supervoxel_clusters.at (adjacent_itr->second);
          PointT pt;
          pt.x=neighbor_supervoxel->centroid_.x;
          pt.y=neighbor_supervoxel->centroid_.y;
          pt.z=neighbor_supervoxel->centroid_.z;
          adjacent_supervoxel_centers.push_back(pt);

          std::stringstream ss;
          ss << "supervoxel_" << supervoxel_label <<"-"<< adjacent_itr->second;
          
          float weight = weight_adjacency.find(
           pair<uint32_t,uint32_t>(label_itr->first,adjacent_itr->second))->second;

          weight = weight;
          if( weight >= 0 )
            addSupervoxelConnectionToViewer(pt_ctr,pt,weight+0.01,ss.str(),viewer);

          //if(weight<1)
          {
              PointT pt_mid;
              pt_mid.x = (pt.x + pt_ctr.x)/2;
              pt_mid.y = (pt.y + pt_ctr.y)/2;
              pt_mid.z = (pt.z + pt_ctr.z)/2;
              std::stringstream sss;
              sss << weight;
              ss << "text";
//              viewer.addText3D(sss.str(), pt_mid, 0.002, 1,0,0, ss.str());
          }
        }
        
        //Move iterator forward to next label
        label_itr = label_adjacency.upper_bound (supervoxel_label);
    }
}

void addSupervoxelIndexToViewer(
    map <uint32_t, Supervoxel<PointXYZRGB>::Ptr > supervoxel_clusters,
    pcl::visualization::PCLVisualizer &viewer
)
{
    for( map <uint32_t, Supervoxel<PointXYZRGB>::Ptr >::iterator it
          = supervoxel_clusters.begin();
         it != supervoxel_clusters.end(); it++ )
    {
        PointXYZRGBA pt = it->second->centroid_;
        std::stringstream ss;
        ss << "[" << it->first << "]";
        viewer.addText3D(ss.str(), pt, 0.002, 0,1,0, ss.str());
    }
}

static vector<Vec3b> colors;
void drawLabels(cv::Mat &input, cv::Mat &dst, int idx_bg)
{
    Mat markers;
    input.convertTo(markers,CV_32SC1);

    double minVal, maxVal;
    minMaxLoc(markers,&minVal,&maxVal);
    int n_colors = (int)maxVal;

    // Generate random colors
    for (size_t i = colors.size(); i < (n_colors+1); i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    // Create the result image
    dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors    
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            //int index = markers.at<int>(i,j);
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= n_colors)
            {
                dst.at<Vec3b>(i,j) = colors[index-1];                
            }
            else
            {
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);                
            }
        }
    }
}