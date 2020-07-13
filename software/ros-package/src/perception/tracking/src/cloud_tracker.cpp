// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/registration_visualizer.h>

// Visualization
#include <pcl/visualization/cloud_viewer.h>
#include "rl_msgs/rl_msgs_conversions.hpp"
#include "rl_msgs/rl_msgs_visualization.hpp"

#include "tracking/cloud_tracker.hpp"
#include "tracking/colors.hpp"

using namespace std;
using namespace cv;
using namespace pcl;

void KNearest2Dto3D::Train( vector<PointCloud<PointXYZRGB>::Ptr>* sv_, 
                            vector<float> &K )
{
    sv = sv_;
    camera_K.resize(9);
    for( int i=0; i<9; i++ ) camera_K[i] = K[i];

    size_t n_cloud = 0;
    for( size_t d=0; d<sv->size(); d++ )
    {
        n_cloud += (*sv)[d]->size();
    }
    Mat pts2d(n_cloud,2,CV_32FC1); // [x,y]
    Mat idxes(n_cloud,1,CV_32FC1);
    labels = Mat(n_cloud,2,CV_32FC1);

    int idx_cloud = 0;
    for( size_t d=0; d<sv->size(); d++ )
    {
        // Create map 2D -> 3D point
        for( size_t p=0; p<(*sv)[d]->size(); p++ )
        {
            PointXYZRGB &pt = (*(*sv)[d])[p];
            float x,y;
            Projection(pt, &x,&y);
            int r=int(y+0.5), c=int(x+0.5);
            
            pts2d.at<float>(idx_cloud,0) = c;
            pts2d.at<float>(idx_cloud,1) = r;
            idxes.at<float>(idx_cloud,0) = idx_cloud;
            labels.at<float>(idx_cloud,0) = d;
            labels.at<float>(idx_cloud,1) = p;
            idx_cloud++;
        }        
    }
    knn->train(pts2d, ml::ROW_SAMPLE, idxes);
}

void KNearest2Dto3D::FindNearest( vector<Point2f>     &pts2d,
                                  vector<PointXYZRGB> &pts3d,
                                  vector<int>         &idxes_sv )
{
    size_t n_pts = pts2d.size();    
    Mat mat2d(n_pts,2,CV_32FC1);
    for( size_t p=0; p<n_pts; p++ )
    {
        mat2d.at<float>(p,0) = pts2d[p].x;
        mat2d.at<float>(p,1) = pts2d[p].y;
    }

    Mat matIDX, dist;
    knn->findNearest(mat2d, 1, noArray(), matIDX, dist);
    pts3d.resize(n_pts);
    idxes_sv.resize(n_pts);
    for( size_t p=0; p<n_pts; p++ )
    {
        if( dist.at<float>(p,0) < 5 )
        {
            int i = (int)matIDX.at<float>(p,0);
            int idx_sv = labels.at<float>(i,0);
            int idx_pt = labels.at<float>(i,1);

            pts3d[p] = (*((*sv)[idx_sv]))[idx_pt];
            idxes_sv[p] = idx_sv;
        }
        else
        {
            pts3d[p].x = 0;
            pts3d[p].y = 0;
            pts3d[p].z = 0;
            idxes_sv[p] = -1;                
        }
    }   
}

void KNearest2Dto3D::Projection(PointXYZRGB &pt, float *x_prj, float *y_prj)
{
    *x_prj = camera_K[0]*pt.x/pt.z + camera_K[2];
    *y_prj = camera_K[4]*pt.y/pt.z + camera_K[5];
}

void CloudTracker::BackProjection( double u, double v, double depth, 
                                   double* x, double* y, double* z  )
{
    depth *= _depth_scale;
    *z = depth;
    *x = (u-_camera_K[2])*depth/_camera_K[0];
    *y = (v-_camera_K[5])*depth/_camera_K[4];
}

typedef pair<size_t,size_t> Vertex; // <image #, id #>
typedef multimap<Vertex,Vertex> Edges;
typedef pair<Edges::iterator,Edges::iterator> EdgeEqualRange;

static 
void DepthFirstTrajectories(
    vector<set<Vertex> > &traj, 
    vector<vector<vector<pair<Point,Point> > > > &point_pairs,
    vector<vector<Point2f> > &points_track,
    int idx, map<Vertex,bool> &visited, Edges &edges, Vertex vertex_cur )
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
            if( it_edge->second.first != vertex_cur.first )
            {
                Point pt_curr
                 = points_track[vertex_cur.first][vertex_cur.second];
                Point pt_next
                 = points_track[it_edge->second.first][it_edge->second.second];

                point_pairs[vertex_cur.first][idx].push_back(
                    pair<Point,Point>(pt_curr,pt_next) );
            }

            DepthFirstTrajectories( traj, point_pairs, points_track,
                                    idx, visited, edges, it_edge->second );
        }
    }    
}

void CloudTracker::BuildTrajectories( 
    vector<PointCloud<PointXYZRGB>::Ptr > &cloud_beg,
    vector<PointCloud<PointXYZRGB>::Ptr > &cloud_end,
    vector<Mat> &images_seg,
    vector<vector<Point2f> > &points_track,
    vector<set<pair<size_t,size_t> > > &traj,
    vector<vector<vector<pair<Point,Point> > > > &point_pairs,
    multimap<size_t,size_t> &beg2end,
    multimap<size_t,size_t> &end2beg    
)
{
    // Find the labels at the first and last frame
    KNearest2Dto3D knn_beg(&cloud_beg, _camera_K);
    KNearest2Dto3D knn_end(&cloud_end, _camera_K);
    
    size_t idx2vld_beg[points_track[0].size()], 
           idx2vld_end[points_track[0].size()];
    vector<Point2f> points_beg, points_end;
    for( size_t p=0; p<points_track[0].size(); p++ )
    {
        Point2f &pt_beg = points_track[0][p];
        Point2f &pt_end = points_track[points_track.size()-1][p];
        if( pt_beg.x >= 0 && pt_beg.y >= 0 )
        {            
            points_beg.push_back(pt_beg);
            idx2vld_beg[p] = points_beg.size()-1;
        }
        else
        {
            idx2vld_beg[p] = -1;
        }

        if( pt_end.x >= 0 && pt_end.y >= 0 )
        {   
            points_end.push_back(pt_end);
            idx2vld_end[p] = points_end.size()-1;
        }
        else
        {
            idx2vld_end[p] = -1;   
        }
    }
    vector<PointXYZRGB> pts3d_beg, pts3d_end;
    vector<int> labels_beg, labels_end;
    knn_beg.FindNearest( points_beg, pts3d_beg, labels_beg );
    knn_end.FindNearest( points_end, pts3d_end, labels_end );

    // Find Trajectories
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
                
                ushort label;
                if( i==0 )
                {
                    int ret = labels_beg[idx2vld_beg[id]];
                    if( ret == -1 ) continue;
                    label = ret; 
                } 
                else
                {
                    label = images_seg[i].at<ushort>(pt_curr.y,pt_curr.x);
                //ushort label = images_seg[i].at<ushort>(pt_curr.y,pt_curr.x);
                } 

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

    point_pairs.resize(n_images-1);
    for( size_t i=0; i<n_images-1; i++ )point_pairs[i].resize(cloud_beg.size());

    // Depth First Labeling    
    traj.resize(cloud_beg.size());
    for( size_t id=0; id<points_track[0].size(); id++ )
    {
        Point pt_curr = points_track[0][id];
        if( pt_curr.x >= 0 && pt_curr.y >= 0 )
        {
            Vertex v_curr(0,id);
            map<Vertex,bool>::iterator it_visited = visited.find(v_curr);
            if( it_visited != visited.end() && 
                it_visited->second == false &&
                edges.count(v_curr)
            )
            {
                int idx = labels_beg[id];
                DepthFirstTrajectories(traj, point_pairs, points_track,
                                       idx, visited, edges, v_curr);                
            }
        }
    }

    // Connect label in the first and last frame    
    for( size_t t=0; t<traj.size(); t++ )
    {
        int label_beg = -1;
        set<int> set_labels_end;
        for( set<Vertex>::iterator it_set  = traj[t].begin(); 
                                   it_set != traj[t].end();   it_set++ )
        {
            if( it_set->first == 0 && label_beg == -1 ) // the vertex in the last frame
            {
                label_beg = labels_beg[idx2vld_beg[it_set->second]];
            }
            if( it_set->first == n_images-1 ) // the vertex in the last frame
            {
                set_labels_end.insert(labels_end[idx2vld_end[it_set->second]]);
            }
            if( label_beg == -1 ) break;
        }
        if( label_beg == -1 ) continue;

        for( set<int>::iterator it_set  = set_labels_end.begin();
                                it_set != set_labels_end.end();   it_set++ )
        {
            if( *it_set == -1 ) continue;
            beg2end.insert(pair<int,int>(label_beg,*it_set));
            end2beg.insert(pair<int,int>(*it_set,label_beg));
        }
    }
}

template <typename PointT>
static bool TransformationEstimationSVD(vector<PointT> &pts1, vector<PointT> &pts2,
    typename registration::TransformationEstimationSVD<PointT,PointT>::Matrix4 &tf)
{
    if( pts1.size() > 3 )
    {
        // get mean point
        PointCloud<PointT> cloud1;
        PointCloud<PointT> cloud2;

        for( size_t p=0; p<pts1.size(); p++ )
        {
            if(pts1[p].z == 0 || pts2[p].z == 0) continue;
            
            cloud1.push_back(pts1[p]);
            cloud2.push_back(pts2[p]);
        }

        registration::TransformationEstimationSVD<PointT,PointT> teSVD;
        teSVD.estimateRigidTransformation (cloud1,cloud2,tf);
    }
    else
    {
        if( pts1.size() > 0 )
        {
            PointXYZ mean_pts1, mean_pts2;
            mean_pts1.x=0; mean_pts1.y=0; mean_pts1.z=0;
            mean_pts2.x=0; mean_pts2.y=0; mean_pts2.z=0;
            for( size_t p=0; p<pts1.size(); p++ )
            {
                mean_pts1.x += pts1[p].x;
                mean_pts1.y += pts1[p].y;
                mean_pts1.z += pts1[p].z;

                mean_pts2.x += pts2[p].x;
                mean_pts2.y += pts2[p].y;
                mean_pts2.z += pts2[p].z;
            }
            mean_pts1.x /= pts1.size();
            mean_pts1.y /= pts1.size();
            mean_pts1.z /= pts1.size();

            mean_pts2.x /= pts2.size();
            mean_pts2.y /= pts2.size();
            mean_pts2.z /= pts2.size();

            tf << 1, 0, 0, mean_pts2.x-mean_pts1.x,
                  0, 1, 0, mean_pts2.y-mean_pts1.y,
                  0, 0, 1, mean_pts2.z-mean_pts1.z,
                  0, 0, 0, 1;
        }
        else
        {
            tf << 1, 0, 0, 0,
                  0, 1, 0, 0,
                  0, 0, 1, 0,
                  0, 0, 0, 1;
            return false;
        }
    }
    return true;
}

void CloudTracker::GetTransformationMatrix( vector<Mat> &depths,
           vector<vector<vector<pair<Point,Point> > > > &point_pairs,
                               vector<vector<Matrix4> > &TFs          )
{
    // Get 3D Points
    size_t n_pairs = point_pairs.size();
    TFs.resize(n_pairs);
    for( size_t i=0; i<n_pairs; i++ )
    {
        size_t n_segs = point_pairs[i].size();
        TFs[i].resize(n_segs);

        for( size_t s=0; s<n_segs; s++ )
        {
            size_t n_points = point_pairs[i][s].size();
            vector<PointXYZ> pts3d_1, pts3d_2;
            for( size_t p=0; p<n_points; p++ )
            {
                Point &pt1 = point_pairs[i][s][p].first;
                Point &pt2 = point_pairs[i][s][p].second;

                double d1 = depths[i  ].at<ushort>(pt1.y,pt1.x);
                double d2 = depths[i+1].at<ushort>(pt2.y,pt2.x);

                double x1,y1,z1, x2,y2,z2;
                BackProjection(pt1.x,pt1.y,d1, &x1,&y1,&z1);
                BackProjection(pt2.x,pt2.y,d2, &x2,&y2,&z2);

                if( z1<0.1 || z2<0.1 ) continue;

                pts3d_1.push_back(PointXYZ(x1,y1,z1));
                pts3d_2.push_back(PointXYZ(x2,y2,z2));
            }            
            TransformationEstimationSVD<PointXYZ>(pts3d_1, pts3d_2, TFs[i][s]);
        }
    }
}

static
void TranformPoint( geometry_msgs::Point &in, CloudTracker::Matrix4 &tf,
                    geometry_msgs::Point &out)
{
    geometry_msgs::Point res;
    res.x = in.x * tf(0,0) + 
            in.y * tf(0,1) + 
            in.z * tf(0,2) + tf(0,3);
    res.y = in.x * tf(1,0) + 
            in.y * tf(1,1) + 
            in.z * tf(1,2) + tf(1,3);
    res.z = in.x * tf(2,0) + 
            in.y * tf(2,1) + 
            in.z * tf(2,2) + tf(2,3);
    out = res;    
}

static 
void TranformNormal( geometry_msgs::Vector3 &in, CloudTracker::Matrix4 &tf, 
                     geometry_msgs::Vector3 &out)
{
    geometry_msgs::Vector3 res;
    res.x = in.x * tf(0,0) + 
            in.y * tf(0,1) + 
            in.z * tf(0,2);
    res.y = in.x * tf(1,0) + 
            in.y * tf(1,1) + 
            in.z * tf(1,2);
    res.z = in.x * tf(2,0) + 
            in.y * tf(2,1) + 
            in.z * tf(2,2);
    out = res;    
}

static
void AverageQuaternion(vector<Eigen::Quaternionf> &qs, Eigen::Quaternionf &mean)
{
    Eigen::MatrixXf Q(4,qs.size());
    for( size_t q=0; q<qs.size(); q++ )
    {
        Q(0,q)=qs[q].w(); Q(1,q)=qs[q].x(); Q(2,q)=qs[q].y(); Q(3,q)=qs[q].z();
    }

    Eigen::MatrixXf QQt = Q * Q.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(4);
    es.compute(QQt);
    Eigen::MatrixXf axes = es.eigenvectors();
    Eigen::Vector4f vec = axes.col(3); // the vector with the largest eigenvalue
    mean = Eigen::Quaternionf(vec(0), vec(1), vec(2), vec(3));
    mean.normalize();
}

static
void AverageTransformation(vector<CloudTracker::Matrix4> &ins,
                           CloudTracker::Matrix4 &out)
{
    float x=0,y=0,z=0;
    vector<Eigen::Quaternionf> qs;
    for( size_t t=0; t<ins.size(); t++ )
    {
        Eigen::Quaternionf q(ins[t].block<3,3>(0,0));
        q.normalize();
        qs.push_back(q);

        x += ins[t](0,3);
        y += ins[t](1,3);
        z += ins[t](2,3);
    }
    x /= ins.size();
    y /= ins.size();
    z /= ins.size();

    Eigen::Quaternionf q;        
    AverageQuaternion(qs,q);
    out << 0,0,0,0,
           0,0,0,0,
           0,0,0,0,
           0,0,0,1;
    out.block<3,3>(0,0) = q.matrix();
    out(0,3)=x; out(1,3)=y; out(2,3)=z;     
}

void CloudTracker::Track(rl_msgs::SegmentationScene &segscene_beg, 
                         rl_msgs::SegmentationScene &segscene_end,
                         vector<Mat> &depths,
                         vector<Mat> &images_seg,
                         vector<vector<Point2f> > &points_track,
                         vector<set<pair<size_t,size_t> > > &traj,
                         multimap<size_t,size_t> &beg2end,
                         multimap<size_t,size_t> &end2beg,
                         vector<Matrix4> &TFs)
{
    size_t n_images = depths.size();

    size_t n_segs_beg = segscene_beg.objects.size();
    size_t n_segs_end = segscene_end.objects.size();

    vector<PointCloud<PointXYZRGB>::Ptr> sv_beg(n_segs_beg),
                                         sv_end(n_segs_end);
    for( size_t s=0; s<n_segs_beg; s++ )
    {
        sv_beg[s] = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
        fromRLMsg(segscene_beg.objects[s], sv_beg[s]);
    }
    for( size_t s=0; s<n_segs_end; s++ )
    {
        sv_end[s] = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
        fromRLMsg(segscene_end.objects[s], sv_end[s]);
    }

    ROS_INFO_STREAM("Track using 2D sift");
    vector<vector<vector<pair<Point,Point> > > > point_pairs;
    BuildTrajectories( sv_beg, sv_end, images_seg, points_track, traj,
                       point_pairs, beg2end, end2beg);

    vector<vector<Matrix4> > TFs_is;
    GetTransformationMatrix(depths, point_pairs, TFs_is);

    TFs.resize(n_segs_beg); // TF = TF_2 * TF_1
    vector<Matrix4> TFs_1(n_segs_beg);
    for( size_t s=0; s<n_segs_beg; s++ )
    {
        Matrix4 TF = Eigen::Affine3f::Identity().matrix();
        for( size_t i=0; i<n_images-1; i++ )
        {
            TF = TF * TFs_is[i][s];
        }
        TFs_1[s] = TF;
    }
    ROS_INFO_STREAM("Track using 2D sift - Done");

    ROS_INFO_STREAM("Track using ICP (fine-tuning)");
    vector<Matrix4> TFs_2(n_segs_beg);
    for( size_t s1=0; s1<n_segs_beg; s1++ )
    {
        vector<Matrix4> TFs_mean;        

        rl_msgs::SegmentationObject &rl_so1
         = segscene_beg.objects[s1];        
        for( size_t f1=0; f1<rl_so1.faces.size(); f1++ )
        {
            rl_msgs::SegmentationFace &rl_fc1 = rl_so1.faces[f1];

            geometry_msgs::Vector3 normal1;
            TranformNormal(rl_fc1.normal,TFs_1[s1],normal1);

            geometry_msgs::Point center1;
            TranformPoint(rl_fc1.center,TFs_1[s1],center1);

            float min_dist = INFINITY;
            int min_s2 = -1;
            int min_f2 = -1;

            // for the candidate segments in the end frame
            pair< multimap<size_t,size_t>::iterator,
                  multimap<size_t,size_t>::iterator > ret
                   = beg2end.equal_range(s1);
            for( multimap<size_t,size_t>::iterator
                 it_beg2end=ret.first; it_beg2end!=ret.second; it_beg2end++ )
            {
                size_t s2 = it_beg2end->second;
                rl_msgs::SegmentationObject &rl_sv2
                 = segscene_end.objects[s2];
                for( size_t f2=0; f2<rl_sv2.faces.size(); f2++ )
                {
                    rl_msgs::SegmentationFace &rl_fc2 = rl_sv2.faces[f2];
           
                    float n = (normal1.x*rl_fc2.normal.x) +
                              (normal1.y*rl_fc2.normal.y) +
                              (normal1.z*rl_fc2.normal.z);
                    if( n < 0.7 ) continue; // significantly different normal

                    for( size_t p2=0; p2<rl_fc2.voxel_points.size(); p2++ )
                    {
                        float dist
                         = (rl_fc2.voxel_points[p2].x - center1.x) *
                           (rl_fc2.voxel_points[p2].x - center1.x) +
                           (rl_fc2.voxel_points[p2].y - center1.y) *
                           (rl_fc2.voxel_points[p2].y - center1.y) +
                           (rl_fc2.voxel_points[p2].z - center1.z) *
                           (rl_fc2.voxel_points[p2].z - center1.z);
                        dist = sqrt(dist);

                        if( min_dist > dist )
                        {
                            min_dist = dist;
                            min_s2 = s2;
                            min_f2 = f2;
                        } 
                    }
                }
            }

            if( min_dist == INFINITY ) continue;

            PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
            fromROSMsg<PointXYZRGB>(rl_fc1.cloud, *cloud);

            PointCloud<PointXYZRGB>::Ptr cloud_fc1(new PointCloud<PointXYZRGB>);
            transformPointCloud(*cloud,*cloud_fc1,TFs_1[s1]);

            PointCloud<PointXYZRGB>::Ptr cloud_fc2(new PointCloud<PointXYZRGB>);
            fromROSMsg<PointXYZRGB>(
              segscene_end.objects[min_s2].faces[min_f2].cloud, *cloud_fc2);

            IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
            icp.setRANSACOutlierRejectionThreshold(0.10);
            icp.setMaxCorrespondenceDistance(0.10);            
            icp.setInputSource(cloud_fc1);
            icp.setInputTarget(cloud_fc2);
            PointCloud<PointXYZRGB>::Ptr cloud_tf(new PointCloud<PointXYZRGB>);
            icp.align(*cloud_tf);
            TFs_mean.push_back(icp.getFinalTransformation());
        }
        if( TFs_mean.size()>1 )
        { 
            AverageTransformation(TFs_mean, TFs_2[s1]);
        }
        else if( TFs_mean.size()==1 )
        {
            TFs_2[s1] = TFs_mean[0];
        }
        else 
        {
            TFs_2[s1] << 1,0,0,0,
                         0,1,0,0,
                         0,0,1,0,
                         0,0,0,1;
        }
        TFs[s1] = TFs_2[s1]*TFs_1[s1];
    }
    ROS_INFO_STREAM("Track using ICP (fine-tuning) - Done");
}