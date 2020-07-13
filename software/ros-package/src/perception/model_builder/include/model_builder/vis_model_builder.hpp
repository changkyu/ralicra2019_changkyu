#ifndef VIS_MODEL_BUILDER__HPP__
#define VIS_MODEL_BUILDER__HPP__

#include <vector>
#include <queue>
#include <time.h>
#include <opencv2/core/core.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/visualization/cloud_viewer.h>

#include "segmentation/seg_lccp_2Dseg.hpp"
#include "tracking/image_tracker.hpp"
#include "tracking/cloud_tracker.hpp"

#include "bullet_simulation/bullet_simulation.hpp"

#include <shape_msgs/Plane.h>
#include "rl_msgs/SegmentationObject.h"

#define NUM_OF_HYPS (32)

typedef class VisModelBuilder
{
public:
    typedef shape_msgs::Plane Plane; // ax + by+ cz - d = 0

    typedef enum ENUM_Criteria
    {
        C_NONE               = 0x0,
        C_GRAVITY            = 0x1,
        C_MANHATTAN_XY       = 0x2,
        C_MANHATTAN          = C_GRAVITY | C_MANHATTAN_XY,
        C_ONEFACE            = 0x4,
        C_TWOFACES           = 0x8 | C_ONEFACE,
        MASK_CRITERIA        = 0xF
    } ENUM_Criteria;

    typedef enum ENUM_GenerateFaceType
    {        
        GEN_MINIMUM=0x1,   GEN_MAXIMUM_H=0x2, GEN_MAXIMUM_V=0x3, 
        GEN_RANDOM=0x4,    GEN_MEDIUM=0x5,   MASK_GENSIZE=0xF,        

        NO_NEWFACE=0x10,      GEN_GRAVFACE=0x20,
        GEN_NEWFACE=0x30,     GEN_NEWFACE_MID=0x40, GEN_NEWFACE_MAX=0x50,
        MASK_NEWFACE=0xF0
    } ENUM_GenerateFaceType;

    typedef struct VisConvexHull
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_hull;
        pcl::PolygonMesh polymesh;
        double cx,cy,cz;
        float likelihood;
        float likelihood_face;
        float penalty_var;
        CloudTracker::Matrix4 tf_cur2init;
    } VisConvexHull;

    typedef struct PointXYZUncertain
    {
        PCL_ADD_POINT4D
        uint8_t gravity[32]; // up to 32 objects
        uint8_t n_face_prjs[32];
        uint8_t type[32];
        int     idxOwner;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    } PointXYZUncertain;

private:
    typedef enum ENUM_Type
    {
        EMPTY=0, HIDDEN, OCCUPIED
    } ENUM_Type;

    typedef struct VisFace
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_prj;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_hull;        
        pcl::PolygonMesh polymesh;
        float area;

        pcl::PointIndices::Ptr inliers;
        pcl::ModelCoefficients::Ptr coefficients;
        Plane plane;
        std::vector<size_t> idxes_uncertain;
    } VisFace;

    typedef struct VisModel
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
        std::vector<VisFace> faces;
        std::vector<VisFace> faces_hidden;        
        std::vector<VisConvexHull> chulls;
        bool tobeUpdated;        
        CloudTracker::Matrix4 tf;
    } VisModel;

#define DEBUG_SIMULATOR 0
#if DEBUG_SIMULATOR
    typedef class VisModelSimulator : public BulletSimulationGui
#else
    typedef class VisModelSimulator : public BulletSimulation
#endif
    {
    public:
        VisModelSimulator(){}
        ~VisModelSimulator(){}

        void SetupStaticScene( Plane &plane_ground,
            const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &clouds
             = std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>(), 
            const std::vector<VisConvexHull*> &chulls
             = std::vector<VisConvexHull*>()                                  );
        float Test( VisConvexHull &vischull, bool show=false );
        float Test( std::vector<VisConvexHull*> &vischulls, bool show=false);

    } VisModelSimulator;

public:
    VisModelBuilder( ros::NodeHandle* nh,
                     std::vector<float> &camera_K,
                     std::vector<float> camera_RT=std::vector<float>(),
                     std::string param="compute_face=true",
                     int width=600, int height=480,
                     float depth_scale=0.000124987 )
    {
        _camera_K.resize(camera_K.size());
        for( int i=0; i<camera_K.size(); i++ ) _camera_K[i] = camera_K[i];

        SetCameraRT(camera_RT);        

        _imgTracker = new ImageTracker(width,height);
        _cloudSeg = new SegLCCP2DSeg(nh);
        _cloudTracker = new CloudTracker(camera_K,depth_scale);

        _cloud_uncertain
         = pcl::PointCloud<PointXYZUncertain>::Ptr(
             new pcl::PointCloud<PointXYZUncertain>);

        _param = param;
    };
    ~VisModelBuilder()
    {
        delete _imgTracker;
        delete _cloudSeg;
        delete _cloudTracker;
    };

    void Update( cv::Mat &image, cv::Mat &depth,
                 std::vector<float> camera_RT=std::vector<float>() );
    void Update( cv::Mat &image, cv::Mat &depth, 
                 pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                 std::vector<float> camera_RT=std::vector<float>() );

    void SetCameraRT(const std::vector<float> &camera_RT);
    
    size_t NumOfModels(){ return _models.size(); }    
    size_t NumOfRemovedPlanes(){ return _planes_bg.size(); }
    Plane GetRemovedPlane(size_t p){ return _planes_bg[p]; }    
    std::vector<Plane>& GetRemovedPlanes();

    void GenerateModel(size_t o, VisConvexHull &vischull, int flag=GEN_RANDOM|GEN_NEWFACE);
    void GenerateModels(VisConvexHull vischulls[][NUM_OF_HYPS]);
    bool GenerateGravityModel(size_t o, VisConvexHull &vischull);
    bool GenerateGravityModel2(size_t o, VisConvexHull &vischull);
    void GenerateTwoFacesModel(size_t o, VisConvexHull &vischull);
    void GetConvexHull(size_t o, ENUM_Criteria criteria, VisConvexHull &chull);
    
    void AddBGPlane(const Plane &plane);
    void AddGroundPlane(const Plane &plane);
    void AddStaticBoxOpened(const std::vector<float> &box);
    
    std::vector<Plane>& GetBGPlanes(){ return _planes_bg; }

    void Sampling(VisConvexHull vischulls[][NUM_OF_HYPS],
        std::vector<std::pair<float,std::vector<VisConvexHull*> > > &res,
        std::vector<std::vector<std::pair<float,VisConvexHull*> > > &probs,
        std::vector<VisConvexHull*> &best, 
        std::vector<VisConvexHull*> &most_stable);

    void Sampling2(
        std::vector<std::vector<std::pair<float,VisConvexHull*> > > &probs,
        std::vector<VisConvexHull*> &best, 
        std::vector<VisConvexHull*> &most_stable);

    void GetSegmentation( int idx, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

    void DrawSegmentation( int idx, pcl::visualization::PCLVisualizer &viewer, 
                           int v1=-1, int v2=-1 );

private:

    void GenerateUncertainVoxels( pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cam,
                                  pcl::PointCloud<PointXYZUncertain> &cloud_uncertain_all );

    void UpdateUncertainVoxels(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cam);

    void AddBGPlane(rl_msgs::SegmentationScene &segscene);
    void AddVisModel(rl_msgs::SegmentationScene &segscene);
    void UpdateVisModel( rl_msgs::SegmentationScene &segscene,
                         std::multimap<size_t,size_t> &end2beg,
                         std::vector<CloudTracker::Matrix4> &TFs    );
    void GenerateFace( pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                       VisFace &face);
        
    void Sampling( VisConvexHull vischulls[][NUM_OF_HYPS], 
                   std::vector<double> prob_objects,
                   std::vector<std::pair<size_t,size_t> > &idxes_chosen,                   
                   std::pair<float,std::vector<VisConvexHull*> >   &res  );
    
    typedef enum ENUM_RandomMirrorType
    {
        MINIMUM_SHAPE=0x1, MAXIMUM_SHAPE=0x2, RANDOM_SHAPE=0x3,   
        GRAVITY_SHAPE=0x4, MEDIUM_SHAPE=0x5,  MASK_SHAPE=0xF,
        FROM_ALL=0x10,     FROM_ONEFACE=0x20, FROM_TWOFACES=0x30, MASK_FROM=0xF0
    } ENUM_RandomMirrorType;

    bool GetRandomMirrorPoints( 
        uint32_t o,
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_mid, 
        Plane plane,
        VisFace &face,
        pcl::PointCloud<PointXYZUncertain>::Ptr cloud_uncertain,
        pcl::search::KdTree<pcl::PointXYZRGB> &kdtree_others,
        int flag, float min_dist=0.02, 
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_other=NULL );

    void UpdateFaces( std::vector<rl_msgs::SegmentationFace> &rl_faces,
                      std::vector<VisFace> &faces,
                      const CloudTracker::Matrix4 &tf=Eigen::Affine3f::Identity().matrix());

    bool IsSameFace(VisFace &f1, VisFace &f2);
    void transform( VisFace &face_in, VisFace &face_out, 
                    const CloudTracker::Matrix4 &tf );
    void transform( VisConvexHull &vchull_in, VisConvexHull &vchull_out, 
                    const CloudTracker::Matrix4 &tf );

    ImageTracker* _imgTracker;
    SegLCCP2DSeg* _cloudSeg;
    CloudTracker* _cloudTracker;
    
    std::vector<float> _camera_K;
    std::vector<float> _camera_RT;
    std::vector<pcl::PointXYZ> _cameras_pos;
    CloudTracker::Matrix4 _tf_cam2world;
    CloudTracker::Matrix4 _tf_world2cam;
    std::string _param;

    std::vector<cv::Mat> _depths;
    std::vector<cv::Mat> _images_seg;

    //std::queue<rl_msgs::SegmentationScene> _segscenes;
    std::vector<rl_msgs::SegmentationScene> _segscenes;

    std::mutex mtx_update_2d;
    std::mutex mtx_update_3d;

    std::vector<VisModel> _models;
    Plane _plane_ground;
    std::vector<Plane> _planes_bg;
    std::vector<Plane> _planes_bg_cam;
        
    std::map<size_t,size_t> _seg2model;
    std::vector<bool> _models_up2date;

    pcl::PointCloud<PointXYZUncertain>::Ptr _cloud_uncertain;
    pcl::search::KdTree<PointXYZUncertain> _kdtree_uncertain;
    const double _resolution_uncertain=0.005f;

    std::multimap<uint32_t,uint32_t> _obj_edges;

    //VisModelSimulator _simulator;

} VisModelBuilder;

inline VisModelBuilder::ENUM_Criteria operator|(
    VisModelBuilder::ENUM_Criteria a, VisModelBuilder::ENUM_Criteria b)
{
    return (VisModelBuilder::ENUM_Criteria)( (int)(a) | (int)(b) );
}
inline VisModelBuilder::ENUM_Criteria& operator|=(
    VisModelBuilder::ENUM_Criteria& a, VisModelBuilder::ENUM_Criteria b)
{
    return (VisModelBuilder::ENUM_Criteria&)( (int&)(a) |= (int)(b) );
}
inline VisModelBuilder::ENUM_Criteria operator&(
    VisModelBuilder::ENUM_Criteria a, VisModelBuilder::ENUM_Criteria b)
{
    return (VisModelBuilder::ENUM_Criteria)( (int)(a) & (int)(b) );
}


#endif