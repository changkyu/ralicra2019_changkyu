#include <pcl/common/geometry.h>

#include "segmentation/graph/gen_graph.hpp"

using namespace std;
using namespace pcl;

void ConnectEdges(PointCloud<PointXYZRGBL> &cloud_seg,
                  set<pair<uint32_t,uint32_t> > &edges )
{   
    for( int x=0; 0 <= x && x < cloud_seg.width; )
    {        
        for( int y=0; 0 <= y && y < cloud_seg.height; )
        {                    
            PointXYZRGBL* ptl = &(cloud_seg.at(x,y));
            PointXYZRGBL* ptl_next;

            int i;
            ptl_next = NULL;
            for( i=1; i<10 && 0 <= x+i && x+i < cloud_seg.width && 
                              0 <= y   && y   < cloud_seg.height;   i++ )
            {
                ptl_next = &(cloud_seg.at(x+i,y));
                if( ptl_next->z != 0 ) 
                {
                    if( ptl_next->label != ptl->label && 
                        geometry::distance( *ptl, *ptl_next ) < 0.003 )
                    {
                        edges.insert(pair<uint32_t,uint32_t>(ptl->label,ptl_next->label));
                        edges.insert(pair<uint32_t,uint32_t>(ptl_next->label,ptl->label));
                    }
                    break;
                }
            }            

            int j;
            ptl_next = NULL;
            for( j=1; j<10 && 0 <= x   && x   < cloud_seg.width && 
                              0 <= y+j && y+j < cloud_seg.height;   j++ )
            {
                ptl_next = &(cloud_seg.at(x,y+j));
                if( ptl_next->z != 0 )
                {
                    if( ptl_next->label != ptl->label && 
                        geometry::distance( *ptl, *ptl_next ) < 0.003 )
                    {
                        edges.insert(pair<uint32_t,uint32_t>(ptl->label,ptl_next->label));
                        edges.insert(pair<uint32_t,uint32_t>(ptl_next->label,ptl->label));
                    }
                    break;
                }
            }
            y = y + j;
        }
        x++;
    }    
}