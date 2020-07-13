#include <iostream>

#include "tracking/seg_tracker.hpp"
#include <pcl/visualization/cloud_viewer.h>

using namespace std;
using namespace cv;
using namespace pcl;

typedef pair<size_t,size_t> Vertex; // <image #, id #>
typedef multimap<Vertex,Vertex> Edges;
typedef pair<Edges::iterator,Edges::iterator> EdgeEqualRange;

static void DepthFirstTrajectories(vector<set<Vertex> > &traj, int idx,
                            map<Vertex,bool> &visited,
                            Edges &edges,
                            Vertex vertex_cur         )
{
    map<Vertex,bool>::iterator it_visited = visited.find(vertex_cur);

    if ( it_visited->second ) return;    
    else it_visited->second = true;

    if( edges.count(vertex_cur) > 0 )
    {
        traj[idx].insert(vertex_cur);
        EdgeEqualRange ret = edges.equal_range(vertex_cur);    
        for(Edges::iterator it_edge  = ret.first; 
                            it_edge != ret.second; it_edge++ )
        {
            DepthFirstTrajectories(traj, idx, visited, edges, it_edge->second);        
        }
    }    
}

void SegTrajectories(vector<Mat> &images_seg, 
                     vector<vector<Point2f> > &points_track,
                     vector<set<Vertex> > &traj             )
{
    Edges edges;
    map<Vertex,bool> visited;
    size_t n_images = images_seg.size();
    
    for( size_t i=0; i<n_images; i++ )
    {
        map<ushort,size_t> seg2pt;
        size_t n_points = points_track[i].size();
        for( size_t id=0; id<n_points; id++ )
        {
            Point pt_curr = points_track[i][id];
            if( pt_curr.x >= 0 && pt_curr.y >= 0 )
            {
                // Bind points in the same segmentation in an image
                Vertex v_curr(i,id);
                visited.insert(pair<Vertex,bool>(v_curr,false));

                ushort label = images_seg[i].at<ushort>(pt_curr.y,pt_curr.x);
                map<ushort,size_t>::iterator it_seg2pt = seg2pt.find(label);
                if( it_seg2pt == seg2pt.end() )
                {
                    seg2pt.insert(pair<ushort,size_t>(label,id));
                }
                else
                {
                    Vertex v_1st(i,it_seg2pt->second);
                    edges.insert(pair<Vertex,Vertex>(v_curr,v_1st ));
                    edges.insert(pair<Vertex,Vertex>(v_1st, v_curr));
                }

                // Bind points in different frames
                if( i<n_images-1 )
                {
                    Point pt_next = points_track[i+1][id];
                    if( pt_next.x >= 0 && pt_next.y >= 0 )
                    {
                        Vertex v_curr(i  ,id);
                        Vertex v_next(i+1,id);    
                        edges.insert(pair<Vertex,Vertex>(v_curr,v_next));
                        //edges.insert(pair<Vertex,Vertex>(v_next,v_curr));            
                    }
                }
            }
        }
    }

    // Depth First Labeling
    int idx=0;
    traj.clear();
    for(map<Vertex,bool>::iterator it_visited  = visited.begin();
                                   it_visited != visited.end();   it_visited++ )
    {
        if( it_visited->second==false && edges.count(it_visited->first) > 0 )
        {
            traj.push_back(set<Vertex>());
            DepthFirstTrajectories(traj, idx, visited, edges, it_visited->first);
            idx++;
        }        
    }    
}
