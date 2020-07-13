#ifndef IMAGE_TRACKER__HPP__
#define IMAGE_TRACKER__HPP__

#include <vector>
#include <memory>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "SiftGPU.h"

typedef struct SiftFeat
{
    std::vector<float> descs;
    std::vector<SiftGPU::SiftKeypoint> keypts;
} SiftFeat;

typedef class ImageTracker
{

public:
    ImageTracker(int n_cols, int n_rows, 
                 int device_id=0, int vec_size=4, int num_iter_ransac=1000);
    ~ImageTracker();

    void AddImage( cv::Mat &image );
    void Track( std::vector<std::vector<cv::Point2f> > &points_track );

private:

    void GetFeatureVectorSIFT(const cv::Mat &image, 
                              std::vector<float> &feats,
                              std::vector<SiftGPU::SiftKeypoint> &keypts);

    void GetFeatureVectorSIFT(const cv::Mat &image, SiftFeat &siftfeat)
    {
      GetFeatureVectorSIFT(image, siftfeat.descs, siftfeat.keypts);
    }
    
    void MatchSIFT( SiftFeat &siftfeat_1,
                    SiftFeat &siftfeat_2,
                    std::vector<cv::Point2f> &pts_match_1,
                    std::vector<cv::Point2f> &pts_match_2
                  );

    // SIFT
    const std::unique_ptr<SiftGPU> _siftEngine;
    const std::unique_ptr<SiftMatchGPU> _siftMatchEngine;
    const int _DESCRIPTOR_LENGTH;
    const int _num_iter_ransac;
    std::vector<int> _match_buffer;
    const int _max_matches;
    int _n_cols, _n_rows;

    std::vector<SiftFeat> _siftfeats;
    std::vector<std::map<std::pair<int,int>,uint32_t> > _pt2id;
    int _id_new;

    std::mutex mtx;

} ImageTracker;

#endif