#include <memory>
#include <iostream>
#include <string.h>
#include <GL/gl.h>
#include <Eigen/Eigen>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

#include "tracking/image_tracker.hpp"

ImageTracker::ImageTracker(int n_cols, int n_rows, 
                       int device_id,int vec_size,int num_iter_ransac) :
    _n_cols{n_cols},
    _n_rows{n_rows},
    _num_iter_ransac{num_iter_ransac},
    _max_matches{50000},
    _DESCRIPTOR_LENGTH{128},
    _siftEngine{std::unique_ptr<SiftGPU>{new SiftGPU()}},
    _siftMatchEngine{
        std::unique_ptr<SiftMatchGPU>{new SiftMatchGPU(4096*vec_size)} }
{
    // SIFT
    char device_id_str[10];
    sprintf(device_id_str, "%d", device_id);
    const char *argv_template[] = {"-m",          "-fo",   "-1",    "-s",
                                   "-v",          "0",     "-pack", "-cuda",
                                    device_id_str, "-maxd", "4608"};
    int argc = sizeof(argv_template) / sizeof(char *);
    char *argv[argc];
    for (int i = 0; i < argc; i++) argv[i] = strdup(argv_template[i]);

    _siftEngine->ParseParam(argc, argv); 

    for (int i = 0; i < argc; i++) free(argv[i]);

    if (_siftEngine->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
        throw std::runtime_error("D_MultipleRigidPoseSparse::D_"
                                 "MultipleRigidPoseSparse: SiftGPU cannot create "
                                 "GL contex\n");
    
    _siftMatchEngine->VerifyContextGL();
    _match_buffer.resize(_max_matches * 2);

    _id_new = 0;
}

ImageTracker::~ImageTracker()
{


}

void ImageTracker::GetFeatureVectorSIFT(const cv::Mat &image, 
                                  std::vector<float> &feats,
                                  std::vector<SiftGPU::SiftKeypoint> &keypts)
{
    if( image.channels() == 1 )
    {
        _siftEngine->RunSIFT(_n_cols, _n_rows, image.data,
                         GL_LUMINANCE, GL_UNSIGNED_BYTE);
        int n_feats = _siftEngine->GetFeatureNum();
        feats.resize(n_feats * _DESCRIPTOR_LENGTH);
        keypts.resize( n_feats);
        _siftEngine->GetFeatureVector(keypts.data(), feats.data());
    }
    else
    {
        // Image Pre-processing
        Mat image_gray;
        cvtColor( image, image_gray, COLOR_RGB2GRAY );
        if( image_gray.cols != _n_cols || image_gray.cols != _n_rows )
        {
            resize(image_gray, image_gray, Size(_n_cols,_n_rows));
        }

        _siftEngine->RunSIFT(_n_cols, _n_rows, image_gray.data,
                         GL_LUMINANCE, GL_UNSIGNED_BYTE);
        int n_feats = _siftEngine->GetFeatureNum();
        feats.resize(n_feats * _DESCRIPTOR_LENGTH);
        keypts.resize( n_feats);
        _siftEngine->GetFeatureVector(keypts.data(), feats.data());
    }
}

void ImageTracker::MatchSIFT( SiftFeat &siftfeat_1,
                              SiftFeat &siftfeat_2,
                              std::vector<cv::Point2f> &pts_match_1,
                              std::vector<cv::Point2f> &pts_match_2
                             )
{
    std::vector<float>* p_feats_1 = &(siftfeat_1.descs);
    std::vector<float>* p_feats_2 = &(siftfeat_2.descs);
    std::vector<SiftGPU::SiftKeypoint>* p_keypts_1 = &(siftfeat_1.keypts);
    std::vector<SiftGPU::SiftKeypoint>* p_keypts_2 = &(siftfeat_2.keypts);
    int n_feats_1 = (int)siftfeat_1.keypts.size();
    int n_feats_2 = (int)siftfeat_2.keypts.size();

    std::vector<float> feats;
    std::vector<SiftGPU::SiftKeypoint> keypts;

    // shuffling and downsampling image keypoints
    const unsigned long long one_d_texture_limit = 134217728;
    if (static_cast<unsigned long long>(n_feats_1) *
            static_cast<unsigned long long>(n_feats_2) >
        one_d_texture_limit) 
    {
        int* p_n_feats;
        if( n_feats_1 > n_feats_2 )
        {   
            p_n_feats = &n_feats_1;
            feats = siftfeat_1.descs;
            keypts  = siftfeat_1.keypts;         
            p_feats_1 = &feats;
            p_keypts_1  = &keypts;
        }
        else
        {
            p_n_feats = &n_feats_2;
            feats = siftfeat_2.descs;
            keypts  = siftfeat_2.keypts;         
            p_feats_2 = &feats;
            p_keypts_2  = &keypts;
        }

        int max_n_image_features = n_feats_1 < n_feats_2?
            one_d_texture_limit / n_feats_1 - 1 :
            one_d_texture_limit / n_feats_2 - 1 ;      

        std::vector<int> shuffle_inds(*p_n_feats);
        std::iota(shuffle_inds.begin(), shuffle_inds.end(), 0);
        std::random_shuffle(shuffle_inds.begin(), shuffle_inds.end());

        Eigen::Map<Eigen::VectorXi> shuffle_inds_eig(shuffle_inds.data(),
                                                     *p_n_feats);
        Eigen::Map<Eigen::MatrixXf> all_keypoints_eig(
          (float *)keypts.data(), 4, *p_n_feats);
        Eigen::Map<Eigen::MatrixXf> all_descriptors_eig(
          feats.data(), _DESCRIPTOR_LENGTH, *p_n_feats);

        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> p(
                                                             shuffle_inds_eig);
        all_keypoints_eig = all_keypoints_eig * p;
        all_descriptors_eig = all_descriptors_eig * p;

        *p_n_feats = max_n_image_features;
    }

    _siftMatchEngine->SetDescriptors(0, n_feats_1,
                                   p_feats_1->data()); // model
    _siftMatchEngine->SetDescriptors(1, n_feats_2,
                                   p_feats_2->data()); // image

    // match and get resultImageTracker
    int num_match = _siftMatchEngine->GetSiftMatch(
        _max_matches, (int(*)[2])_match_buffer.data());

    pts_match_1.clear();
    pts_match_2.clear();
    for(int i=0; i<num_match; i++)
    {
        SiftGPU::SiftKeypoint keypts_1 = (*p_keypts_1)[_match_buffer.at(i*2)];
        SiftGPU::SiftKeypoint keypts_2 = (*p_keypts_2)[_match_buffer.at(i*2+1)];

        pts_match_1.push_back(cv::Point2f(keypts_1.x,keypts_1.y));
        pts_match_2.push_back(cv::Point2f(keypts_2.x,keypts_2.y));
    }
}

void ImageTracker::AddImage( Mat &image )
{
    mtx.lock();

    SiftFeat siftfeat;
    GetFeatureVectorSIFT(image, siftfeat);
    _siftfeats.push_back(siftfeat);    
    _pt2id.push_back(map<pair<int,int>,uint32_t>());

    if( _siftfeats.size() > 1 )
    {
        int i=_siftfeats.size()-2;

        vector<Point2f> pts_match_src;
        vector<Point2f> pts_match_dst;

        MatchSIFT( _siftfeats[i], _siftfeats[i+1],
                   pts_match_src, pts_match_dst    );

        for( size_t p=0; p<pts_match_src.size(); p++ )
        {
            double dist = norm(pts_match_src[p]-pts_match_dst[p]);
            if( dist > 100 ) continue; // outlier

            pair<int,int> key_src(pts_match_src[p].x,pts_match_src[p].y);
            pair<int,int> key_dst(pts_match_dst[p].x,pts_match_dst[p].y);
            map<pair<int,int>,uint32_t>::iterator it_pt2id
             = _pt2id[i].find(key_src);
            
            if( it_pt2id == _pt2id[i].end() )
            {
                _pt2id[i  ].insert(pair<pair<int,int>,uint32_t>(key_src,_id_new));
                _pt2id[i+1].insert(pair<pair<int,int>,uint32_t>(key_dst,_id_new));
                _id_new++;
            }
            else
            {
                uint32_t id = it_pt2id->second;
                _pt2id[i+1].insert(pair<pair<int,int>,uint32_t>(key_dst,id));
            }            
        }
    }
    else
    {
        _id_new = 0;
    }

    mtx.unlock();
}

void ImageTracker::Track( vector<vector<Point2f> > &points_track )
{
    mtx.lock();

    size_t n_images = _siftfeats.size();
    points_track.resize(n_images);
    for( size_t i=0; i<n_images; i++ )
    {
        points_track[i].resize(_id_new);
        for( size_t id=0; id<_id_new; id++ )
        {
            points_track[i][id].x = -1;
            points_track[i][id].y = -1;
        }
        for( map<pair<int,int>,uint32_t>::iterator it_pt2id = _pt2id[i].begin();
             it_pt2id != _pt2id[i].end(); it_pt2id++                    )
        {
            Point pt(it_pt2id->first.first, it_pt2id->first.second);
            points_track[i][it_pt2id->second] = pt;
        }
    }

    for( size_t i=0; i<n_images; i++ )
    {
        SiftFeat siftfeat = _siftfeats.back();
        _siftfeats.clear();
        _siftfeats.push_back(siftfeat);

        _pt2id.clear();        
    }

    mtx.unlock();    
}
