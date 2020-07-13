#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Geometry> 

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include <sensor_msgs/CameraInfo.h>
#include <tf/LinearMath/Transform.h>
#include <tf/transform_datatypes.h>
#include "rl_msgs/tracking_image_srv.h"
#include "rl_msgs/tracking_cloud_srv.h"
#include "rl_msgs/seg_supervoxels_srv.h"
#include "rl_msgs/rl_msgs_visualization.hpp"

#include "tracking/image_tracker.hpp"

using namespace std;
using namespace cv;
using namespace pcl;

typedef struct KNN2to3
{
    Ptr<ml::KNearest> knn;
    Mat labels; // [idx_supervoxel, idx_point]
} KNN2to3;

typedef registration
        ::TransformationEstimationSVD<PointXYZRGB,PointXYZRGB>
        ::Matrix4 Matrix4;

ros::NodeHandle* nh;
Mat camera_K_default(3,3,CV_32FC1);
template<typename PointT>
void Projection_K(PointT &pt, Mat &K, 
                  float *x_prj, float *y_prj)
{
    *x_prj = K.at<float>(0,0)*pt.x/pt.z + K.at<float>(0,2);
    *y_prj = K.at<float>(1,1)*pt.y/pt.z + K.at<float>(1,2);    
}

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

bool flag = true;
int idx_show=1;
int idx_mode=1;
void Callback_pclkeyboard (const pcl::visualization::KeyboardEvent &event,
                           void* viewer_void)
{
    if (event.keyDown())
    {        
        if( event.getKeySym()=="Escape" ||
            event.getKeyCode()=='Q'     ||
            event.getKeyCode()=='q'         )
        {
            flag = false;
        }
        else if( event.getKeyCode()=='+' )
        {
            idx_show = idx_show + 1;
        }
        else if( event.getKeyCode()=='-' )
        {
            idx_show = idx_show - 1;;
        }
        else if( event.getKeyCode()=='*' )
        {
            idx_show = idx_show + 2;;
        }        
        else if( event.getKeyCode()=='/' )
        {
            idx_show = idx_show - 2;;
        }
        else if( event.getKeyCode()=='1' )
        {
            idx_mode = 1;
        }
        else if( event.getKeyCode()=='2' )
        {
            idx_mode = 2;
        }
        else if( event.getKeyCode()=='3' )
        {
            idx_mode = 3;
        }
        else if( event.getKeyCode()=='4' )
        {
            idx_mode = 4;
        }
    }
}

#define IMAGE_WIDTH  (600)
#define IMAGE_HEIGHT (480)
ImageTracker _imgTracker(IMAGE_WIDTH, IMAGE_HEIGHT);

void toROSMsg(Matrix4 &tf, geometry_msgs::Transform &msg_tf)
{
    msg_tf.translation.x = tf(0,3);
    msg_tf.translation.x = tf(1,3);
    msg_tf.translation.x = tf(2,3);

    tf::Matrix3x3 tf3d;
    tf3d.setValue(tf(0,0), tf(0,1), tf(0,2), 
                  tf(1,0), tf(1,1), tf(1,2), 
                  tf(2,0), tf(2,1), tf(2,2));

    tf::Quaternion tfqt;
    tf3d.getRotation(tfqt);
    quaternionTFToMsg(tfqt,msg_tf.rotation);
}

bool Tracking_sift_image(vector<sensor_msgs::Image> &msg_images,
                         vector<sensor_msgs::RegionOfInterest> &regions,
                         vector<vector<vector<Point2f> > > &pts_match_src,
                         vector<vector<vector<Point2f> > > &pts_match_dst,
                         vector<double> &homographies    )
{
    size_t n_images = msg_images.size();

    Mat images[n_images];
    vector<SiftFeat> siftfeats(n_images);    
    for( size_t i=0; i<n_images; i++ )
    {
        // Get Input Image
        cv_bridge::CvImagePtr cv_ptr        
         = cv_bridge::toCvCopy(msg_images[i],sensor_msgs::image_encodings::BGR8);
        cv_ptr->image.copyTo(images[i]);

        // Image Pre-processing
        Mat image_gray;
        cvtColor( cv_ptr->image, image_gray, COLOR_RGB2GRAY );
        if( image_gray.cols != IMAGE_WIDTH || image_gray.cols != IMAGE_HEIGHT )
        {
            resize(image_gray, image_gray, Size(IMAGE_WIDTH,IMAGE_HEIGHT));
        }
        
        // Extract SIFT features
        _imgTracker.GetFeatureVectorSIFT(image_gray, siftfeats[i]);
    }

    // Interesting Regions in the initial frame
    vector<vector<Rect> > rects(n_images);    
    vector<vector<Rect> > rects_ex(n_images);    
    for( size_t r=0; r<regions.size(); r++ )
    {
        rects[0].push_back(
            Rect( regions[r].x_offset,
                  regions[r].y_offset,
                  regions[r].width   ,
                  regions[r].height    ) );

        rects_ex[0].push_back(
            Rect( regions[r].x_offset - regions[r].width *0.1,
                  regions[r].y_offset - regions[r].height*0.1,
                  regions[r].width    + regions[r].width *0.2,
                  regions[r].height   + regions[r].height*0.2 ) );
    }

    // Track the regions
    pts_match_src.resize(n_images-1);
    pts_match_dst.resize(n_images-1);
    for( size_t i=0; i<n_images-1; i++ )
    {        
        vector<Point2f> pts_match_srcimg;
        vector<Point2f> pts_match_dstimg;
        _imgTracker.MatchSIFT( siftfeats[i].descs,   siftfeats[i].keypts,
                               siftfeats[i+1].descs, siftfeats[i+1].keypts,
                               pts_match_srcimg, pts_match_dstimg           );
        
        // SIFT features in the intersting regions
        pts_match_src[i].resize(regions.size());
        pts_match_dst[i].resize(regions.size());
        for( size_t r=0; r<regions.size(); r++ )
        {
            // filter out not interesting region
            vector<Point2f> &pts_match_src_local = pts_match_src[i][r];
            vector<Point2f> &pts_match_dst_local = pts_match_dst[i][r];
            pts_match_src_local.clear(); 
            pts_match_dst_local.clear();

            for( size_t k=0; k<pts_match_srcimg.size(); k++ )
            {
                if(rects_ex[i][r].contains(pts_match_srcimg[k]))
                {
                    pts_match_src_local.push_back(pts_match_srcimg[k]);
                    pts_match_dst_local.push_back(pts_match_dstimg[k]);
                }
            }

            Mat H(3,3,CV_64FC1);
            if( pts_match_src_local.size() > 3 )
            {
                // Compute Homography Matrix
                H = findHomography( pts_match_src_local,
                                    pts_match_dst_local, CV_RANSAC );
            }
            else
            {
                float d_x=0, d_y=0;
                for( size_t p=0; p<pts_match_src_local.size(); p++ )
                {
                    d_x += pts_match_dst_local[p].x - pts_match_src_local[p].x;
                    d_y += pts_match_dst_local[p].y - pts_match_src_local[p].y;
                }
                if( pts_match_src_local.size() > 0 )
                {
                    d_x /= pts_match_src_local.size();
                    d_y /= pts_match_src_local.size();
                }

                H.at<double>(0,0)=1;H.at<double>(0,1)=0;H.at<double>(0,2)=d_x;
                H.at<double>(1,0)=0;H.at<double>(1,1)=1;H.at<double>(1,2)=d_y;
                H.at<double>(2,0)=0;H.at<double>(2,1)=0;H.at<double>(2,2)=1;
            }

            for( int a=0; a<H.rows; a++ )
            {
                for( int b=0; b<H.cols; b++ )
                {
                    homographies.push_back(H.at<double>(a,b));
                }
            }

            // Track the region
            Rect &rect = rects[i][r];
            vector<Point2f> pts_src(4);
            pts_src[0].x = rect.x;            pts_src[0].y = rect.y;
            pts_src[1].x = rect.x+rect.width; pts_src[1].y = rect.y;
            pts_src[2].x = rect.x+rect.width; pts_src[2].y = rect.y+rect.height;
            pts_src[3].x = rect.x;            pts_src[3].y = rect.y+rect.height;

            vector<Point2f> pts_dst;
            perspectiveTransform(pts_src, pts_dst, H);

            float min_x= INFINITY, min_y= INFINITY, 
                  max_x=-INFINITY, max_y=-INFINITY;
            for( size_t p=0; p<pts_dst.size(); p++ )
            {
                if( min_x > pts_dst[p].x ) min_x = pts_dst[p].x;
                if( min_y > pts_dst[p].y ) min_y = pts_dst[p].y;
                if( max_x < pts_dst[p].x ) max_x = pts_dst[p].x;
                if( max_y < pts_dst[p].y ) max_y = pts_dst[p].y;
            }

            Rect rect_next(min_x, min_y, max_x-min_x, max_y-min_y);
            rects[i+1].push_back(rect_next);
            rects_ex[i+1].push_back(
                Rect(rect_next.x      - rect_next.width *0.1,
                     rect_next.y      - rect_next.height*0.1,
                     rect_next.width  + rect_next.width *0.2,
                     rect_next.height + rect_next.height*0.2 ));
        }        
    }
    return true;
}

bool Tracking_sift_image(rl_msgs::tracking_image_srv::Request  &req, 
                         rl_msgs::tracking_image_srv::Response &res)
{
    ROS_INFO_STREAM("Track SIFT image");
    vector<vector<vector<Point2f> > > pts_match_src, pts_match_dst;
    bool ret = Tracking_sift_image( req.images,      req.regions, 
                                    pts_match_src,   pts_match_dst, 
                                    res.homographies               );
    ROS_INFO_STREAM("Track SIFT image - Done");
    return ret;
}

void TrainKNearest2dto3d(vector<PointCloud<PointXYZRGB>::Ptr> &sv,
                         Mat &camera_K,
                         KNN2to3 &knn2to3                         )
{
    size_t n_cloud = 0;
    for( size_t d=0; d<sv.size(); d++ )
    {
        n_cloud += sv[d]->size();
    }
    Mat pts2d(n_cloud,2,CV_32FC1); // [x,y]
    Mat idxes(n_cloud,1,CV_32FC1);
    knn2to3.labels = Mat(n_cloud,2,CV_32FC1);
    int idx_cloud = 0;

    for( size_t d=0; d<sv.size(); d++ )
    {
        // Create map 2D -> 3D point
        for( size_t p=0; p<sv[d]->size(); p++ )
        {
            PointXYZRGB &pt = (*sv[d])[p];
            float x,y;
            Projection_K<PointXYZRGB>(pt,camera_K, &x,&y);
            int r=int(y+0.5), c=int(x+0.5);
            
            pts2d.at<float>(idx_cloud,0) = c;
            pts2d.at<float>(idx_cloud,1) = r;
            idxes.at<float>(idx_cloud,0) = idx_cloud;
            knn2to3.labels.at<float>(idx_cloud,0) = d;
            knn2to3.labels.at<float>(idx_cloud,1) = p;
            idx_cloud++;
        }        
    }
    knn2to3.knn->train(pts2d, ml::ROW_SAMPLE, idxes);
}

void FindKNearest2dto3d(vector<PointCloud<PointXYZRGB>::Ptr> &sv,
                        KNN2to3 &knn2to3,
                        vector<vector<Point2f> >     &pts2d,
                        vector<vector<PointXYZRGB> > &pts3d,
                        vector<vector<int> >         &idxes_sv   )
{
    size_t n_pts = 0;
    for( size_t s=0; s<pts2d.size(); s++ )
    {
        n_pts += pts2d[s].size();
    }
    
    int idx = 0;
    Mat mat2d(n_pts,2,CV_32FC1);
    for( size_t s=0; s<pts2d.size(); s++ )
    {
        for( size_t p=0; p<pts2d[s].size(); p++ )
        {
            mat2d.at<float>(idx,0) = pts2d[s][p].x;
            mat2d.at<float>(idx,1) = pts2d[s][p].y;
            idx++;
        }        
    }

    idx = 0;
    Mat matIDX, dist;
    knn2to3.knn->findNearest(mat2d, 1, noArray(), matIDX, dist);
    pts3d.resize(sv.size());
    idxes_sv.resize(sv.size());
    for( size_t s=0; s<sv.size(); s++ )
    {
        pts3d[s].resize(pts2d[s].size());
        idxes_sv[s].resize(pts2d[s].size());
        for( size_t p=0; p<pts2d[s].size(); p++ )
        {
            if( dist.at<float>(idx,0) < 5 )
            {
                int i = (int)matIDX.at<float>(idx,0);
                int idx_sv = knn2to3.labels.at<float>(i,0);
                int idx_pt = knn2to3.labels.at<float>(i,1);

                pts3d[s][p] = (*(sv[idx_sv]))[idx_pt];
                idxes_sv[s][p] = idx_sv;
            }
            else
            {
                pts3d[s][p].x = 0;
                pts3d[s][p].y = 0;
                pts3d[s][p].z = 0;
                idxes_sv[s][p] = -1;                
            }
            idx++;
        }
    }
}

void FindKNearest2dto3d(vector<PointCloud<PointXYZRGB>::Ptr> &sv,
                        KNN2to3             &knn2to3,
                        vector<Point2f>     &pts2d,
                        vector<PointXYZRGB> &pts3d,
                        vector<int>         &idxes_sv   )
{
    size_t n_pts = pts2d.size();    
    Mat mat2d(n_pts,2,CV_32FC1);
    for( size_t p=0; p<n_pts; p++ )
    {
        mat2d.at<float>(p,0) = pts2d[p].x;
        mat2d.at<float>(p,1) = pts2d[p].y;
    }        
    Mat matIDX, dist;
    knn2to3.knn->findNearest(mat2d, 1, noArray(), matIDX, dist);

    pts3d.resize(n_pts);
    idxes_sv.resize(n_pts);
    for( size_t p=0; p<n_pts; p++ )
    {
        if( dist.at<float>(p,0) < 5 )
        {
            int i = (int)matIDX.at<float>(p,0);
            int idx_sv = knn2to3.labels.at<float>(i,0);
            int idx_pt = knn2to3.labels.at<float>(i,1);

            pts3d[p] = (*(sv[idx_sv]))[idx_pt];
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

template<typename PointT>
static
bool TransformationEstimationSVD(vector<PointT> &pts1, vector<PointT> &pts2,
    typename registration::TransformationEstimationSVD<PointT,PointT>::Matrix4 &tf)
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

    if( cloud1.size() > 3 )
    {
        registration::TransformationEstimationSVD<PointT,PointT> teSVD;
        teSVD.estimateRigidTransformation (cloud1,cloud2,tf);
    }
    else
    {
        if( cloud1.size() > 0 )
        {
            PointXYZRGB mean_pts1, mean_pts2;
            mean_pts1.x=0; mean_pts1.y=0; mean_pts1.z=0;
            mean_pts2.x=0; mean_pts2.y=0; mean_pts2.z=0;
            for( size_t p=0; p<cloud1.size(); p++ )
            {
                mean_pts1.x += cloud1[p].x;
                mean_pts1.y += cloud1[p].y;
                mean_pts1.z += cloud1[p].z;

                mean_pts2.x += cloud2[p].x;
                mean_pts2.y += cloud2[p].y;
                mean_pts2.z += cloud2[p].z;
            }
            mean_pts1.x /= cloud1.size();
            mean_pts1.y /= cloud1.size();
            mean_pts1.z /= cloud1.size();

            mean_pts2.x /= cloud2.size();
            mean_pts2.y /= cloud2.size();
            mean_pts2.z /= cloud2.size();

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

void Tracking_sift_image(vector<sensor_msgs::Image> &msg_images,
                         vector<PointCloud<PointXYZRGB>::Ptr>* sv_clouds,
                         KNN2to3* knn2to3,
                         vector<Matrix4>* TFs,
                         vector<bool>* succs_TFs,
                         vector<set<int> >*  idxsv_track                )
{
    size_t n_images = msg_images.size();

    // Extract SIFT features in the images
    Mat images[n_images];
    vector<SiftFeat> siftfeats(n_images);
    for( size_t i=0; i<n_images; i++ )
    {
        // Get Input Image
        cv_bridge::CvImagePtr cv_ptr
         = cv_bridge::toCvCopy(msg_images[i],sensor_msgs::image_encodings::BGR8);
        cv_ptr->image.copyTo(images[i]);

        // Image Pre-processing
        Mat image_gray;
        cvtColor( cv_ptr->image, image_gray, COLOR_RGB2GRAY );
        if( image_gray.cols != IMAGE_WIDTH || image_gray.cols != IMAGE_HEIGHT )
        {
            resize(image_gray, image_gray, Size(IMAGE_WIDTH,IMAGE_HEIGHT));
        }
        
        // Extract SIFT features
        _imgTracker.GetFeatureVectorSIFT(image_gray, siftfeats[i]);
    }

    // Find Matches    
    for( size_t i=0; i<n_images-1; i++ )
    {        

        vector<Point2f> pts_match_src, pts_match_dst;
        _imgTracker.MatchSIFT( siftfeats[i].descs,   siftfeats[i].keypts,
                               siftfeats[i+1].descs, siftfeats[i+1].keypts,
                               pts_match_src,        pts_match_dst         );

        // 2D -> 3D
        vector<PointXYZRGB> pts3d_src, pts3d_dst;
        vector<int> idxes_sv_src, idxes_sv_dst;
        FindKNearest2dto3d( sv_clouds[i],   knn2to3[i],   pts_match_src,
                            pts3d_src,      idxes_sv_src                );
        FindKNearest2dto3d( sv_clouds[i+1], knn2to3[i+1], pts_match_dst,
                            pts3d_dst,      idxes_sv_dst                );
        
        // Find valid matching points
        vector<vector<PointXYZRGB> > sv_pts3d_src(sv_clouds[i].size()), 
                                     sv_pts3d_dst(sv_clouds[i].size());
        int n_occurs[sv_clouds[i].size()];
        for( size_t s=0; s<sv_clouds[i].size(); s++ ) n_occurs[s]=0;

        map<pair<int,int>,int> sd2occur;
        for( size_t p=0; p<pts3d_src.size(); p++ )
        {
            int s = idxes_sv_src[p], d = idxes_sv_dst[p];
            if( s >= 0 && d >= 0 )
            {
                sv_pts3d_src[s].push_back(pts3d_src[p]);
                sv_pts3d_dst[s].push_back(pts3d_dst[p]);
                
                pair<int,int> key(s,d);
                map<pair<int,int>,int>::iterator it_sd2occur = sd2occur.find(key);
                if( it_sd2occur == sd2occur.end() )
                {                    
                    sd2occur.insert(pair<pair<int,int>,int>(key,1));
                }
                else
                {
                    it_sd2occur->second++;
                }
                n_occurs[s]++;
            }
        }
        idxsv_track[i].resize(sv_clouds[i].size());
        for( map<pair<int,int>,int>::iterator it_sd2occur = sd2occur.begin();
             it_sd2occur != sd2occur.end(); it_sd2occur++ )
        {
            int s = it_sd2occur->first.first;
            //if( it_sd2occur->second > n_occurs[s]/3 )
            {
                idxsv_track[i][s].insert(it_sd2occur->first.second);
            }
        }        

        // Find Transformation Matrix
        TFs[i].resize(sv_clouds[i].size());
        succs_TFs[i].resize(sv_clouds[i].size());
        for( size_t s=0; s<sv_clouds[i].size(); s++ )
        {
            succs_TFs[i][s] = TransformationEstimationSVD<PointXYZRGB>(
                                   sv_pts3d_src[s], sv_pts3d_dst[s], TFs[i][s]);
        }
    }
}

ros::ServiceClient clt_seg;
bool Tracking_sift_icp_cloud(rl_msgs::tracking_cloud_srv::Request &req,
                             rl_msgs::tracking_cloud_srv::Response &res )
{
    const size_t n_scenes = req.images.size();

    ROS_INFO_STREAM("PointCloud2 -> pcl::PointCloud");
    PointCloud<PointXYZRGB>::Ptr clouds[n_scenes];
    vector<PointCloud<PointXYZRGB>::Ptr> sv_clouds[n_scenes];
    for( size_t i=0; i<n_scenes; i++ )
    {
        clouds[i] = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
        for( size_t s=0; s<req.segscenes[i].supervoxels.size(); s++ )
        {
            PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
            fromROSMsg<PointXYZRGB>(req.segscenes[i].supervoxels[s].cloud,*cloud);
            sv_clouds[i].push_back(cloud);
            *(clouds[i]) += *cloud;
        }
    }
    ROS_INFO_STREAM("PointCloud2 -> pcl::PointCloud - Done");

    ROS_INFO_STREAM("KNN 2 to 3");    
    KNN2to3 knn2to3[n_scenes];
    for( size_t i=0; i<n_scenes; i++ )
    {
        Mat camera_K(3,3,CV_32FC1);
        if( req.segscenes[i].camera_K.size()>0 )
        {
            camera_K = Mat(3,3,CV_32FC1,req.segscenes[i].camera_K.data());
        }
        else
        {
            camera_K = camera_K_default;
        }        
        knn2to3[i].knn = ml::KNearest::create();
        TrainKNearest2dto3d(sv_clouds[i], camera_K, knn2to3[i]);
    }    
    ROS_INFO_STREAM("KNN 2 to 3 - Done");
    
    ROS_INFO_STREAM("Find Transformation using SIFT");    
    vector<Matrix4> TFs[n_scenes-1];
    vector<bool> succs_TFs[n_scenes-1];
    vector<set<int> > idxsv_track[n_scenes-1];
    Tracking_sift_image(req.images, sv_clouds, knn2to3, TFs, succs_TFs, idxsv_track);
    ROS_INFO_STREAM("Find Transformation using SIFT - Done");

    res.transforms.resize(n_scenes-1);
    for( size_t i=0; i<n_scenes-1; i++ )
    {
        res.transforms[i].tf.resize(req.segscenes[i].supervoxels.size());
        for( size_t s=0; s<req.segscenes[i].supervoxels.size(); s++ )
        {
            res.transforms[i].idxes_source.push_back(s);
            
            for( set<int>::iterator it_set = idxsv_track[i][s].begin();
                 it_set != idxsv_track[i][s].end(); it_set++           )
            {
                res.transforms[i].idxes_target.push_back((uint32_t)*it_set);
            }

            res.transforms[i].tf[s].matrix4x4.resize(16);
            res.transforms[i].tf[s].matrix4x4[0]  = TFs[i][s](0,0);
            res.transforms[i].tf[s].matrix4x4[1]  = TFs[i][s](0,1);
            res.transforms[i].tf[s].matrix4x4[2]  = TFs[i][s](0,2);
            res.transforms[i].tf[s].matrix4x4[3]  = TFs[i][s](0,3);
            res.transforms[i].tf[s].matrix4x4[4]  = TFs[i][s](1,0);
            res.transforms[i].tf[s].matrix4x4[5]  = TFs[i][s](1,1);
            res.transforms[i].tf[s].matrix4x4[6]  = TFs[i][s](1,2);
            res.transforms[i].tf[s].matrix4x4[7]  = TFs[i][s](1,3);
            res.transforms[i].tf[s].matrix4x4[8]  = TFs[i][s](2,0);
            res.transforms[i].tf[s].matrix4x4[9]  = TFs[i][s](2,1);
            res.transforms[i].tf[s].matrix4x4[10] = TFs[i][s](2,2);
            res.transforms[i].tf[s].matrix4x4[11] = TFs[i][s](2,3);
            res.transforms[i].tf[s].matrix4x4[12] = TFs[i][s](3,0);
            res.transforms[i].tf[s].matrix4x4[13] = TFs[i][s](3,1);
            res.transforms[i].tf[s].matrix4x4[14] = TFs[i][s](3,2);
            res.transforms[i].tf[s].matrix4x4[15] = TFs[i][s](3,3);
        }        
    }

////////////////////////////
    visualization::PCLVisualizer viewer;
    viewer.setSize(600,480);
    viewer.setPosition(600,0);        
    viewer.setCameraPosition(0,0,-1,0,0,1,0,-1,0);
    viewer.registerKeyboardCallback(Callback_pclkeyboard);    

    double colors_show[100][n_scenes][3];
    PointCloud<PointXYZRGB>::Ptr clouds_show[n_scenes*2+1];
    for( size_t i=0; i<n_scenes; i++ )
    {
        ROS_INFO_STREAM(2*i);
        ROS_INFO_STREAM(2*i+1);

        clouds_show[2*i]  =PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
        clouds_show[2*i+1]=PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
    }
    ROS_INFO_STREAM(n_scenes*2);
    clouds_show[n_scenes*2]=PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
    for( size_t i=0; i<n_scenes-1; i++ )
    {
        // base i=0
        if( i==0 )
        {
            for( size_t s=0; s<sv_clouds[i].size(); s++ )
            {
                colors_show[s][i][0] = colors[s][0];
                colors_show[s][i][1] = colors[s][1];
                colors_show[s][i][2] = colors[s][2];

                for( size_t p=0; p<sv_clouds[i][s]->size(); p++ )
                {
                    PointXYZRGB pt;
                    pt.x = (*sv_clouds[i][s])[p].x;
                    pt.y = (*sv_clouds[i][s])[p].y;
                    pt.z = (*sv_clouds[i][s])[p].z;
                    pt.r = colors[s][0]; 
                    pt.g = colors[s][1];
                    pt.b = colors[s][2];
                    clouds_show[2*i]->push_back(pt);
                }
            }
        }

        // set [i+1] based on [i]
        for( size_t s=0; s<sv_clouds[i].size(); s++ )
        {
            for( set<int>::iterator
                 it_idxsv_track =idxsv_track[i][s].begin(); 
                 it_idxsv_track!=idxsv_track[i][s].end();  it_idxsv_track++)
            {
                int s_next = *it_idxsv_track;
                colors_show[s_next][i+1][0] = colors_show[s][i][0];
                colors_show[s_next][i+1][1] = colors_show[s][i][1];
                colors_show[s_next][i+1][2] = colors_show[s][i][2];

                for( size_t p=0; p<sv_clouds[i+1][s_next]->size(); p++ )
                {
                    PointXYZRGB pt;
                    pt.x = (*sv_clouds[i+1][s_next])[p].x;
                    pt.y = (*sv_clouds[i+1][s_next])[p].y;
                    pt.z = (*sv_clouds[i+1][s_next])[p].z;

                    pt.r = colors_show[s_next][i+1][0];
                    pt.g = colors_show[s_next][i+1][1];
                    pt.b = colors_show[s_next][i+1][2];
                    clouds_show[2*(i+1)]->push_back(pt);
                }
            }
        }

        // tran
        for( size_t s=0; s<sv_clouds[i].size(); s++ )
        {
            PointCloud<PointXYZRGB>::Ptr cloud_tran(new PointCloud<PointXYZRGB>);
            transformPointCloud(*(sv_clouds[i][s]),*cloud_tran,TFs[i][s]);
            
            for( size_t p=0; p<cloud_tran->size(); p++ )
            {
                PointXYZRGB pt;
                pt.x = (*cloud_tran)[p].x;
                pt.y = (*cloud_tran)[p].y;
                pt.z = (*cloud_tran)[p].z;
                pt.r = colors_show[s][i][0];
                pt.g = colors_show[s][i][1];
                pt.b = colors_show[s][i][2];
                clouds_show[2*i+1]->push_back(pt);
            }
        }
    }

    idx_show = 0;
    int idx_cur = -1;
    int mode_cur = -1;
    while( flag )
    {
        if( idx_show < 0 ) idx_show = 0;
        if( idx_show >= n_scenes*2 ) idx_show = n_scenes*2-1;

        if( idx_cur!=idx_show || mode_cur!=idx_mode )
        {
            ROS_INFO_STREAM("idx:" << idx_show << ", mode:" <<idx_mode);
            viewer.removePointCloud();
            if( idx_mode==1 )
            {
                viewer.addPointCloud(clouds_show[idx_show]);
            }
            else if( idx_mode==2 )
            {
                viewer.addPointCloud(clouds[idx_show/2]);
            }
            idx_cur = idx_show;
            mode_cur = idx_mode;
        }
        viewer.spinOnce(100);
    } 

#if 0
    ROS_INFO_STREAMvisualization::PCLVisualizer viewer;
    viewer.setSize(600,480);
    viewer.setPosition(600,0);        
    viewer.setCameraPosition(0,0,-1,0,0,1,0,-1,0);
    viewer.registerKeyboardCallback(Callback_pclkeyboard);
    viewer.addPointCloud(clouds[0]);

    idx_show = 1;
    int idx_cur = idx_show;
    while( flag )
    {
        if( idx_cur!=idx_show )
        {
            viewer.removePointCloud();
            if( idx_show==1 ) viewer.addPointCloud(clouds[0]);
            if( idx_show==2 ) viewer.addPointCloud(cloud_trans);
            if( idx_show==3 ) viewer.addPointCloud(cloud_trans_final);
            if( idx_show==4 ) viewer.addPointCloud(cloud_targets);            
            else              viewer.addPointCloud(clouds[1]);
            
            idx_cur = idx_show;
        }
        viewer.spinOnce(100);
    }("Find Transformation using ICP");
    // Find 3D transformation    
    PointCloud<PointXYZRGB>::Ptr cloud_targets(new PointCloud<PointXYZRGB>);
    PointCloud<PointXYZRGB>::Ptr cloud_trans(new PointCloud<PointXYZRGB>);
    PointCloud<PointXYZRGB>::Ptr cloud_trans_final(new PointCloud<PointXYZRGB>);
    for( size_t s=0; s<req.supervoxels_src.size(); s++ )    
    {
        PointCloud<PointXYZRGB>::Ptr cloud_target(new PointCloud<PointXYZRGB>);
        for( set<int>::iterator it_idxsv_track = idxsv_track[0][s].begin(); 
             it_idxsv_track != idxsv_track[0][s].end(); it_idxsv_track++     )
        {
            //*cloud_target += *sv_clouds[1][*it_idxsv_track];
            for( size_t p=0; p<sv_clouds[1][*it_idxsv_track]->size(); p++ )
            {
                PointXYZRGB pt;
                pt.x = (*sv_clouds[1][*it_idxsv_track])[p].x;
                pt.y = (*sv_clouds[1][*it_idxsv_track])[p].y;
                pt.z = (*sv_clouds[1][*it_idxsv_track])[p].z;
                pt.r = colors[s][0]; 
                pt.g = colors[s][1];
                pt.b = colors[s][2];
                cloud_target->push_back(pt);
                clouidx_show = 1;
    int idx_cur = idx_show;
    while( flag )
    {
        if( idx_cur!=idx_show )
        {
            viewer.removePointCloud();
            if( idx_show==1 ) viewer.addPointCloud(clouds[0]);
            if( idx_show==2 ) viewer.addPointCloud(cloud_trans);
            if( idx_show==3 ) viewer.addPointCloud(cloud_trans_final);
            if( idx_show==4 ) viewer.addPointCloud(cloud_targets);            
            else              viewer.addPointCloud(clouds[1]);
            
            idx_cur = idx_show;
        }
        viewer.spinOnce(100);
    }d_targets->push_back(pt);
            }
        }

        IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
        //icp.setMaximumIterations(50);
        icp.setRANSACOutlierRejectionThreshold(0.05);
        icp.setMaxCorrespondenceDistance(0.05);
        if( succs_TFs[0][s] )
        {
            PointCloud<PointXYZRGB>::Ptr cloud_tran(new PointCloud<PointXYZRGB>);
            transformPointCloud(*(sv_clouds[0][s]),*cloud_tran,TFs[0][s]);

            icp.setInputSource(cloud_tran);
            icp.setInputTarget(cloud_target);

            PointCloud<PointXYZRGB> cloud_tf;
            icp.align(cloud_tf);

            //*cloud_trans += *cloud_tran;
            for( size_t p=0; p<cloud_tran->size(); p++ )
            {
                PointXYZRGB pt;
                pt.x = (*cloud_tran)[p].x;
                pt.y = (*cloud_tran)[p].y;
                pt.z = (*cloud_tran)[p].z;
                pt.r = colors[s][0]; 
                pt.g = colors[s][1];
                pt.b = colors[s][2];
                cloud_trans->push_back(pt);
                cloud_trans->push_back(pt);
            }

            //*cloud_trans_final += cloud_tf;
            for( size_t p=0; p<cloud_tf.size(); p++ )
            {
                PointXYZRGB pt;
                pt.x = cloud_tf[p].x;
                pt.y = cloud_tf[p].y;
                pt.z = cloud_tf[p].z;
                pt.r = colors[s][0]; 
                pt.g = colors[s][1];
                pt.b = colors[s][2];
                cloud_trans_final->push_back(pt);
                cloud_trans_final->push_back(pt);
            }
        }
        else
        {
            icp.setInputSource(sv_clouds[0][s]);
            icp.setInputTarget(cloud_target);

            PointCloud<PointXYZRGB> cloud_tf;
            icp.align(cloud_tf);

            *cloud_trans += *sv_clouds[0][s];
            *cloud_trans_final += cloud_tf;
        }
    }
    ROS_INFO_STREAM("Find Transformation using SIFT and ICP - Done");

//////////////////////////////
    visualization::PCLVisualizer viewer;
    viewer.setSize(600,480);
    viewer.setPosition(600,0);        
    viewer.setCameraPosition(0,0,-1,0,0,1,0,-1,0);
    viewer.registerKeyboardCallback(Callback_pclkeyboard);
    viewer.addPointCloud(clouds[0]);

    idx_show = 1;
    int idx_cur = idx_show;
    while( flag )
    {
        if( idx_cur!=idx_show )
        {
            viewer.removePointCloud();
            if( idx_show==1 ) viewer.addPointCloud(clouds[0]);
            if( idx_show==2 ) viewer.addPointCloud(cloud_trans);
            if( idx_show==3 ) viewer.addPointCloud(cloud_trans_final);
            if( idx_show==4 ) viewer.addPointCloud(cloud_targets);            
            else              viewer.addPointCloud(clouds[1]);
            
            idx_cur = idx_show;
        }
        viewer.spinOnce(100);
    }
    
    return true;

#endif
#if 0    
    vector<vector<PointXYZRGB> > pts3d_src;
    FindKNearest2dto3d(sv_cloud_src, *knn2to3_src, labels_src, pts_match_src[0], 
                       pts3d_src );
    vector<vector<PointXYZRGB> > pts3d_dst;
    FindKNearest2dto3d(sv_cloud_dst, *knn2to3_dst, labels_dst, pts_match_dst[0], 
                       pts3d_dst );    



    ROS_INFO_STREAM("Tracking 2D image");
    // Track the segment region in 2D images
    vector<sensor_msgs::Image> images;
    images.push_back(req.image_src); 
    images.push_back(req.image_dst);    

    // Interesting Regions in the source image
    vector<double> homographies;
    vector<sensor_msgs::RegionOfInterest> regions;
    vector<Point2f> centers_src;
    for( size_t s=0; s<req.supervoxels_src.size(); s++ )
    {
        sensor_msgs::RegionOfInterest &region = req.supervoxels_src[s].region;
        centers_src.push_back(Point2f(region.x_offset + region.width /2.0,
                                      region.y_offset + region.height/2.0));
        regions.push_back(region);
    }
    
    // Track the object in 2D image
    vector<vector<vector<Point2f> > > pts_match_src, pts_match_dst;    
    Tracking_sift_image( images,        regions, 
                         pts_match_src, pts_match_dst, 
                         homographies                 );
    ROS_INFO_STREAM("Tracking 2D image - Done");

    ROS_INFO_STREAM("Find 2D -> 3D");


    // KNN 2D -> 3D 
    KNN2to3

    Mat labels_src;
    Ptr<ml::KNearest> knn2to3_src(ml::KNearest::create());
    TrainKNearest2dto3d(sv_cloud_src, *knn2to3_src, labels_src);
    Mat labels_dst;
    Ptr<ml::KNearest> knn2to3_dst(ml::KNearest::create());
    TrainKNearest2dto3d(sv_cloud_dst, *knn2to3_dst, labels_dst);

    vector<vector<PointXYZRGB> > pts3d_src;
    FindKNearest2dto3d(sv_cloud_src, *knn2to3_src, labels_src, pts_match_src[0], 
                       pts3d_src );
    vector<vector<PointXYZRGB> > pts3d_dst;
    FindKNearest2dto3d(sv_cloud_dst, *knn2to3_dst, labels_dst, pts_match_dst[0], 
                       pts3d_dst );    
    ROS_INFO_STREAM("Find 2D -> 3D - Done");

    ROS_INFO_STREAM("Estimate Transformation");
    typedef registration
            ::TransformationEstimationSVD<PointXYZRGB,PointXYZRGB>
            ::Matrix4 Matrix4;
    bool succs_track[pts3d_src.size()];
    Matrix4 TFs[pts3d_src.size()];
    for( size_t s=0; s<pts3d_src.size(); s++ )
    {
        succs_track[s] = TransformationEstimationSVD<PointXYZRGB>(
                                            pts3d_src[s],pts3d_dst[s],TFs[s]);
        if( succs_track[s] )
        {
            cout << endl << endl;
            cout << TFs[s] << endl;
            for( size_t p=0; p<pts3d_src[s].size(); p++ )
            {
                if( pts3d_src[s][p].z != 0)
                cout << pts3d_src[s][p] << endl;
            }
        }
    }
    ROS_INFO_STREAM("Estimate Transformation - Done");
    
    // Find 3D transformation
    PointCloud<PointXYZRGB>::Ptr cloud_trans(new PointCloud<PointXYZRGB>);
    for( size_t s=0; s<req.supervoxels_src.size(); s++ )
    {
        if( succs_track[s] )
        {
            PointCloud<PointXYZRGB>::Ptr cloud_tran(new PointCloud<PointXYZRGB>);
            transformPointCloud(*(sv_cloud_src[s]),*cloud_tran,TFs[s]);
            *cloud_trans += *cloud_tran;
        }
    }

    //////////////////////////////
    visualization::PCLVisualizer viewer;
    viewer.setSize(600,480);
    viewer.setPosition(600,0);        
    viewer.setCameraPosition(0,0,-1,0,0,1,0,-1,0);
    viewer.registerKeyboardCallback(Callback_pclkeyboard);
    viewer.addPointCloud(cloud_src);

    idx_show = 1;
    int idx_cur = idx_show;
    while( flag )
    {
        if( idx_cur!=idx_show )
        {
            viewer.removePointCloud();
            if( idx_show==1 ) viewer.addPointCloud(cloud_src);
            if( idx_show==2 ) viewer.addPointCloud(cloud_trans);
            else              viewer.addPointCloud(cloud_dst);
            
            idx_cur = idx_show;
        }
        viewer.spinOnce(100);
    }
    
    return true;
#endif
#if 0
    // Find the candidate segmentation pairs
    Mat pts2d_track(req.supervoxels_src.size(),2,CV_32FC1);
    vector<Point2f> centers_track;
    for( size_t s=0; s<req.supervoxels_src.size(); s++ )
    {
        Mat H(3,3,CV_64FC1, homographies.data() + 9*s);

        Point2f pt_track;
        pt_track.x = H.at<double>(0,0)*centers_src[s].x + 
                     H.at<double>(0,1)*centers_src[s].y + H.at<double>(0,2);
        pt_track.y = H.at<double>(1,0)*centers_src[s].x +
                     H.at<double>(1,1)*centers_src[s].y + H.at<double>(1,2);
        float    z = H.at<double>(2,0)*centers_src[s].x + 
                     H.at<double>(2,1)*centers_src[s].y + H.at<double>(2,2);
        pt_track.x /= z; pt_track.y /= z;
        centers_track.push_back(pt_track);

        pts2d_track.at<float>(s,0) = pt_track.x;
        pts2d_track.at<float>(s,1) = pt_track.y;
    }

    Mat idxes_track, dist;
    knn->findNearest(pts2d_track, 1, noArray(), idxes_track, dist);
    vector<int> labels_track(req.supervoxels_src.size());
    for( size_t s=0; s<req.supervoxels_src.size(); s++ )
    {
        if( dist.at<float>(s,0) < 5 )
        {
            int idx = (int)idxes_track.at<float>(s,0);            
            labels_track[s] = label_dst.at<float>(idx,0);
        }
        else
        {
            labels_track[s] = -1;
        }
    }
#endif
    //////////////////////


#if 0
    // Find 3D transformation
    PointCloud<PointXYZRGB>::Ptr cloud1(new PointCloud<PointXYZRGB>);
    PointCloud<PointXYZRGB>::Ptr cloud12(new PointCloud<PointXYZRGB>);
    PointCloud<PointXYZRGB>::Ptr cloud2(new PointCloud<PointXYZRGB>);

    IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
    icp.setMaximumIterations(50);
    icp.setRANSACOutlierRejectionThreshold(0.01);
    icp.setMaxCorrespondenceDistance(0.1);
    for( size_t s=0; s<req.supervoxels_src.size(); s++ )
    {
        if( labels_track[s] > -1 )
        {
            // Prepare cloud
            int d = labels_track[s];
            PointCloud<PointXYZRGB>::Ptr cloud_src(new PointCloud<PointXYZRGB>);
            fromROSMsg<PointXYZRGB>(req.supervoxels_src[s].cloud, *cloud_src);
            PointCloud<PointXYZRGB>::Ptr cloud_dst(new PointCloud<PointXYZRGB>);
            fromROSMsg<PointXYZRGB>(req.supervoxels_dst[s].cloud, *cloud_dst);
            
            Eigen::Affine3f transform(Eigen::Translation3f(
              req.supervoxels_dst[d].center.x-req.supervoxels_src[s].center.x,
              req.supervoxels_dst[d].center.y-req.supervoxels_src[s].center.y,
              req.supervoxels_dst[d].center.z-req.supervoxels_src[s].center.z));

            PointCloud<PointXYZRGB>::Ptr cloud_src2(new PointCloud<PointXYZRGB>);            
            transformPointCloud(*cloud_src, *cloud_src2, transform);
            icp.setInputSource(cloud_src2);
            icp.setInputTarget(cloud_dst);

            PointCloud<PointXYZRGB> cloud_tf;
            icp.align(cloud_tf);

            *cloud1  += *cloud_src;
            *cloud12 += cloud_tf;
            *cloud2  += *cloud_dst;
        }
    }

    //////////////////////////////
    visualization::PCLVisualizer viewer;
    viewer.setSize(600,480);
    viewer.setPosition(600,0);        
    viewer.setCameraPosition(0,0,-1,0,0,1,0,-1,0);
    viewer.registerKeyboardCallback(Callback_pclkeyboard);
    viewer.addPointCloud(cloud1);

    idx_show = 1;
    int idx_cur = idx_show;
    while( flag )
    {
        if( idx_cur!=idx_show )
        {
            viewer.removePointCloud();
            if( idx_show==1 ) viewer.addPointCloud(cloud1);
            if( idx_show==2 ) viewer.addPointCloud(cloud12);
            else              viewer.addPointCloud(cloud2);
            
            idx_cur = idx_show;
        }
        viewer.spinOnce(100);
    }
    
    return true;

#endif

#if 0
    //////////////////////////////
    cv_bridge::CvImagePtr cv_ptr        
     = cv_bridge::toCvCopy(req.image_src,sensor_msgs::image_encodings::BGR8);
    Mat image_src;
    cv_ptr->image.copyTo(image_src);
    for( size_t s=0; s<req.supervoxels_src.size(); s++ )
    {
        Rect rect(req.supervoxels_src[s].region.x_offset, 
                  req.supervoxels_src[s].region.y_offset,
                  req.supervoxels_src[s].region.width, 
                  req.supervoxels_src[s].region.height   );

        Scalar color(colors[s%20][0],colors[s%20][1],colors[s%20][2]);
        rectangle(image_src, rect, color);
        circle(image_src,centers_src[s],3,color,3);

        stringstream ss;
        ss << s;
        putText(image_src, ss.str(), centers_src[s],  FONT_HERSHEY_SIMPLEX, 1, color);
    }    
    imshow("src",image_src);

    cv_bridge::CvImagePtr cv_ptr2     
     = cv_bridge::toCvCopy(req.image_dst,sensor_msgs::image_encodings::BGR8);
    Mat image_dst;
    cv_ptr2->image.copyTo(image_dst);
    for( size_t s=0; s<req.supervoxels_src.size(); s++ )
    {
        Scalar color(colors[s%20][0],colors[s%20][1],colors[s%20][2]);
        int d = labels_track[s];
        if( d >= 0 )
        {        
            Rect rect(req.supervoxels_dst[d].region.x_offset, 
                      req.supervoxels_dst[d].region.y_offset,
                      req.supervoxels_dst[d].region.width, 
                      req.supervoxels_dst[d].region.height   );
            rectangle(image_dst, rect, color);
            circle(image_dst,centers_track[s],3,color,3);
            stringstream ss;
            ss << d;
            putText(image_dst, ss.str(), centers_track[s], FONT_HERSHEY_SIMPLEX, 1, color);
        }        
    }    
    imshow("dst",image_dst);
    waitKey();

#endif

#if 0
    cv_bridge::CvImagePtr cv_ptr0
     = cv_bridge::toCvCopy(images[0],sensor_msgs::image_encodings::BGR8);    
    cv_bridge::CvImagePtr cv_ptr1
     = cv_bridge::toCvCopy(images[1],sensor_msgs::image_encodings::BGR8);

    Mat img_show(cv_ptr0->image.rows,cv_ptr0->image.cols+cv_ptr1->image.cols,CV_8UC3);
    cv_ptr0->image.copyTo(img_show(cv::Rect(0,0,cv_ptr0->image.cols,cv_ptr0->image.rows)));
    cv_ptr1->image.copyTo(img_show(cv::Rect(cv_ptr0->image.cols,0,
                                            cv_ptr1->image.cols,cv_ptr1->image.rows)));

    for( int i=0; i<2; i++ )
    {
        for( size_t s=0; s<req.supervoxels_src.size(); s++ )
        {
            for( size_t p=0; p<pts_match_src[0][s].size(); p++ )
            {
                circle(img_show,pts_match_src[0][s][p],1,Scalar(0,255,0));

                Point2f pt_dst;
                pt_dst.x = pts_match_dst[0][s][p].x + cv_ptr0->image.cols;
                pt_dst.y = pts_match_dst[0][s][p].y;
                circle(img_show,pt_dst,1,Scalar(0,255,0));

                line(img_show, pts_match_src[0][s][p], pt_dst, Scalar(255,0,0));
            }
        }        
    }
    imshow("sift",img_show);
    
    visualization::PCLVisualizer viewer;
    viewer.setSize(600,480);
    viewer.setPosition(600,0);        
    viewer.setCameraPosition(0,0,-1,0,0,1,0,-1,0);
    viewer.registerKeyboardCallback(Callback_pclkeyboard);    
    for( size_t s=0; s<req.supervoxels_src.size(); s++ )
    {
        PointCloud<PointXYZRGB>::Ptr cloud_src(new PointCloud<PointXYZRGB>);
        fromROSMsg<PointXYZRGB>(req.supervoxels_src[s].cloud, *cloud_src);

        stringstream ss;
        ss << s << endl;
        viewer.addPointCloud(cloud_src, ss.str() + "_cloud");
        PointXYZ pt;
        pt.x = req.supervoxels_src[s].center.x;
        pt.y = req.supervoxels_src[s].center.y;
        pt.z = req.supervoxels_src[s].center.z;
        viewer.addText3D(ss.str(),pt,0.02, 0,1,0, ss.str() + "_center");
    }

    while(flag)
    {
        waitKey(100);
        viewer.spinOnce(100);
    }
#endif
    return true;

/*
    visualization::PCLVisualizer viewer_src;
    viewer_src.setSize(600,480);
    viewer_src.setPosition(600,0);        
    viewer_src.setCameraPosition(0,0,-1,0,0,1,0,-1,0);
    AddSegmentationSupervoxels(req.supervoxels_src,viewer_src);

    visualization::PCLVisualizer viewer_dst;
    viewer_dst.setSize(600,480);
    viewer_dst.setPosition(600,0);        
    viewer_dst.setCameraPosition(0,0,-1,0,0,1,0,-1,0);
    AddSegmentationSupervoxels(req.supervoxels_src,viewer_dst);
*/


    return true;
}

bool Tracking_fpfh_cloud(rl_msgs::tracking_cloud_srv::Request &req,
                             rl_msgs::tracking_cloud_srv::Request &res )
{
    return true;
}

string SRV_SIFT_IMAGE;
string SRV_SIFT_ICP_CLOUD;
string SRV_FPFH_CLOUD;
string SRV_LCCP_2DSEG;

void ParseParam(ros::NodeHandle nh)
{    
    nh.param<string>("srv_tracking_sift_image",SRV_SIFT_IMAGE,"tracking/sift_image"); 
    nh.param<string>("srv_tracking_sift_icp_cloud",SRV_SIFT_ICP_CLOUD,"tracking/sift_icp_cloud"); 
    nh.param<string>("srv_tracking_fpfh_cloud",SRV_FPFH_CLOUD,"tracking/fpfh_cloud"); 
    nh.param<string>("srv_seg_lccp_2Dseg",SRV_LCCP_2DSEG,"segmentation/lccp_2Dseg"); 
}

int main(int argc, char* argv[])
{
    // ROS
    ros::init(argc,argv,"tracking_sift_image_node");
    nh = new ros::NodeHandle;

    ParseParam(*nh);

    sensor_msgs::CameraInfo::ConstPtr ci_depth 
     = ros::topic::waitForMessage<sensor_msgs::CameraInfo>
         ("/depth/camera_info",*nh,ros::Duration(1));
    
    float* data = (float*)camera_K_default.data;
    if( ci_depth )
    {        
        for( int k=0; k<9; k++ ) data[k] = ci_depth->K[k];
        ROS_INFO_STREAM(endl << "camera_K: " << endl << camera_K_default << endl);
    }
    else
    {
        float K[9] = {615.957763671875,               0.0, 308.10989379882810, 
                                    0.0, 615.9578247070312, 246.33352661132812, 
                                    0.0,               0.0,               1.0 };
        for( int k=0; k<9; k++ ) data[k] = K[k];
        ROS_INFO_STREAM(endl << "camera_K (default): " << endl << camera_K_default << endl);
    }

    ros::ServiceServer srv_sift_image
     = nh->advertiseService(SRV_SIFT_IMAGE, Tracking_sift_image);
    ros::ServiceServer srv_sift_icp_cloud
     = nh->advertiseService(SRV_SIFT_ICP_CLOUD, Tracking_sift_icp_cloud);    
    clt_seg = nh->serviceClient<rl_msgs::seg_supervoxels_srv>(SRV_LCCP_2DSEG);

    ros::spin();

    delete nh;
    return 0;
}
