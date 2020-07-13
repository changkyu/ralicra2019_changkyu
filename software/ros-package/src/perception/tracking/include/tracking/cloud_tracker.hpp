#ifndef CLOUD_TRACKER__HPP__
#define CLOUD_TRACKER__HPP__

#include <vector>
#include <map>
#include <ros/ros.h>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include "rl_msgs/SegmentationScene.h"

typedef class KNearest2Dto3D
{
public:
    KNearest2Dto3D()
    {
        knn = cv::ml::KNearest::create();
    }

    KNearest2Dto3D( std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>* sv, 
                    std::vector<float> &K)
    {
        knn = cv::ml::KNearest::create();
        Train(sv, K);
    }
    ~KNearest2Dto3D(){}

    void Train( std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>* sv_, 
                std::vector<float> &K                                     );

    void FindNearest( std::vector<cv::Point2f>      &pts2d,
                      std::vector<pcl::PointXYZRGB> &pts3d,
                      std::vector<int>              &idxes_sv );
private:
    cv::Ptr<cv::ml::KNearest> knn;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>* sv;
    cv::Mat labels;
    std::vector<float> camera_K;
    void Projection(pcl::PointXYZRGB &pt, float *x_prj, float *y_prj);
    
} KNearest2Dto3D;

typedef class CloudTracker
{
public:
    typedef pcl::registration
        ::TransformationEstimationSVD<pcl::PointXYZRGB,pcl::PointXYZRGB>
        ::Matrix4 Matrix4;
        
    CloudTracker(std::vector<float> &K,
                 float depth_scale=0.000124987) :
    _depth_scale{depth_scale}
    {
        _camera_K.resize(K.size());
        for( int i=0; i<K.size(); i++ ) _camera_K[i] = K[i];
    };

    ~CloudTracker(){};

    void Track( rl_msgs::SegmentationScene &segscene_beg,
                rl_msgs::SegmentationScene &segscene_end,
                std::vector<cv::Mat> &depths,
                std::vector<cv::Mat> &images_seg,
                std::vector<std::vector<cv::Point2f> > &points_track,
                std::vector<std::set<std::pair<size_t,size_t> > > &traj,
                std::multimap<size_t,size_t> &beg2end,
                std::multimap<size_t,size_t> &end2beg,
                std::vector<Matrix4> &TFs
    );

private:
    
    float _depth_scale;
    std::vector<float> _camera_K;
    void BackProjection( double u, double v, double depth, 
                         double* x, double* y, double* z  );

    void BuildTrajectories( 
        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr > &cloud_beg,
        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr > &cloud_end,
        std::vector<cv::Mat> &images_seg,
        std::vector<std::vector<cv::Point2f> > &points_track,
        std::vector<std::set<std::pair<size_t,size_t> > > &traj,
        std::vector<std::vector<std::vector<std::pair<cv::Point,cv::Point> > > > &point_pairs,
        std::multimap<size_t,size_t> &beg2end,
        std::multimap<size_t,size_t> &end2beg
    );

    void GetTransformationMatrix( 
        std::vector<cv::Mat> &depths,
        std::vector<std::vector<std::vector<std::pair<cv::Point,cv::Point> > > > &point_pairs,
        std::vector<std::vector<Matrix4> > &TFs );

} CloudTracker;

#endif