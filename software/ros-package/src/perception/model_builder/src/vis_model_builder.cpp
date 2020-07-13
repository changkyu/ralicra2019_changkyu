#include <thread>

#include <cv_bridge/cv_bridge.h>

#include <pcl/point_types.h>

#include <pcl/impl/instantiate.hpp>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/filters/impl/crop_hull.hpp>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>

#include "segmentation/quickshift/quickshift_wrapper.hpp"
#include "rl_msgs/rl_msgs_conversions.hpp"
#include "rl_msgs/rl_msgs_visualization.hpp"

#include "model_builder/vis_model_builder.hpp"

using namespace std;
using namespace cv;
using namespace pcl;

#define COS_10 (0.984807753012208)
#define COS_15 (0.9659258262890683)
#define COS_30 (0.86602540378443860)
#define COS_45 (0.70710678118654757)
#define COS_60 (0.5)
#define COS_75 (0.25881904510252074)

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

POINT_CLOUD_REGISTER_POINT_STRUCT (VisModelBuilder::PointXYZUncertain,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (uint8_t[32], gravity,     gravity    )
                                   (uint8_t[32], n_face_prjs, n_face_prjs)
                                   (uint8_t[32], type,        type       )
                                   (int,         idxOwner,    idxOwner   )
)

PCL_INSTANTIATE(PCLBase,(VisModelBuilder::PointXYZUncertain));
PCL_INSTANTIATE(VoxelGrid,(VisModelBuilder::PointXYZUncertain));
PCL_INSTANTIATE(CropHull,(VisModelBuilder::PointXYZUncertain));
PCL_INSTANTIATE(KdTree,(VisModelBuilder::PointXYZUncertain));

//template class PCLBase<VisModelBuilder::PointXYZUncertain>;
//template class VoxelGrid<VisModelBuilder::PointXYZUncertain>;

typedef VisModelBuilder::Plane Plane;
//typedef VisModelBuilder::VisModel VisModel;

template<typename PointT>
static
void Projection_K(PointT &pt, vector<float> &K, 
                  float *x_prj, float *y_prj)
{
    *x_prj = K[0]*pt.x/pt.z + K[2];
    *y_prj = K[4]*pt.y/pt.z + K[5];
}

static 
void copyTo(const vector<float> &vec, CloudTracker::Matrix4 &mat)
{
    assert(vec.size()==16);
    mat << vec[0],  vec[1],  vec[2],  vec[3], 
           vec[4],  vec[5],  vec[6],  vec[7], 
           vec[8],  vec[9],  vec[10], vec[11], 
           vec[12], vec[13], vec[14], vec[15];
}

template<typename PointT>
static 
void GetPlane(const Normal &normal, const PointT &pt, Plane &plane)
{
    double norm = sqrt( normal.normal_x*normal.normal_x +
                        normal.normal_y*normal.normal_y +
                        normal.normal_z*normal.normal_z   );

    plane.coef[0] = normal.normal_x / norm;
    plane.coef[1] = normal.normal_y / norm;
    plane.coef[2] = normal.normal_z / norm;
    plane.coef[3] = (plane.coef[0]*pt.x + 
                     plane.coef[1]*pt.y + 
                     plane.coef[2]*pt.z   );
}

template<typename PointT>
static 
void GetPlane(const Eigen::Vector3f &normal, const PointT &pt, Plane &plane)
{
    Normal nm(normal(0),normal(1),normal(2));
    GetPlane<PointT>(nm, pt, plane);
}

static 
void rotatePlane(const Plane &plane_in, Plane &plane_out, const CloudTracker::Matrix4 &tf)
{
    CloudTracker::Matrix4 tf_inv = tf.inverse();
    Plane plane_tmp;
    plane_tmp.coef[0] = plane_in.coef[0] * tf_inv(0,0) + 
                        plane_in.coef[1] * tf_inv(1,0) + 
                        plane_in.coef[2] * tf_inv(2,0) + 
                       -plane_in.coef[3] * tf_inv(3,0); 
    plane_tmp.coef[1] = plane_in.coef[0] * tf_inv(0,1) + 
                        plane_in.coef[1] * tf_inv(1,1) + 
                        plane_in.coef[2] * tf_inv(2,1) + 
                       -plane_in.coef[3] * tf_inv(3,1); 
    plane_tmp.coef[2] = plane_in.coef[0] * tf_inv(0,2) + 
                        plane_in.coef[1] * tf_inv(1,2) + 
                        plane_in.coef[2] * tf_inv(2,2) + 
                       -plane_in.coef[3] * tf_inv(3,2); 
    plane_tmp.coef[3] = plane_in.coef[0] * tf_inv(0,3) + 
                        plane_in.coef[1] * tf_inv(1,3) + 
                        plane_in.coef[2] * tf_inv(2,3) + 
                       -plane_in.coef[3] * tf_inv(3,3); 

    float norm = sqrt( plane_tmp.coef[0]*plane_tmp.coef[0] + 
                       plane_tmp.coef[1]*plane_tmp.coef[1] + 
                       plane_tmp.coef[2]*plane_tmp.coef[2]   );
    
    plane_out.coef[0] = plane_tmp.coef[0] / norm;
    plane_out.coef[1] = plane_tmp.coef[1] / norm;
    plane_out.coef[2] = plane_tmp.coef[2] / norm;
    plane_out.coef[3] =-plane_tmp.coef[3] / norm;
}

static 
void transform(const PointXYZ &pt_in, PointXYZ &pt_out, const CloudTracker::Matrix4 &tf)
{
    PointXYZ pt;
    pt.x = tf(0,0)*pt_in.x + tf(0,1)*pt_in.y + tf(0,2)*pt_in.z + tf(0,3);
    pt.y = tf(1,0)*pt_in.x + tf(1,1)*pt_in.y + tf(1,2)*pt_in.z + tf(1,3);
    pt.z = tf(2,0)*pt_in.x + tf(2,1)*pt_in.y + tf(2,2)*pt_in.z + tf(2,3);
    pt_out = pt;
}

static 
void transform(const Normal &nm_in, Normal &nm_out, const CloudTracker::Matrix4 &tf)
{
    Normal nm;
    nm.normal_x = tf(0,0)*nm_in.normal_x + tf(0,1)*nm_in.normal_y + tf(0,2)*nm_in.normal_z;
    nm.normal_y = tf(1,0)*nm_in.normal_x + tf(1,1)*nm_in.normal_y + tf(1,2)*nm_in.normal_z;
    nm.normal_z = tf(2,0)*nm_in.normal_x + tf(2,1)*nm_in.normal_y + tf(2,2)*nm_in.normal_z;
    nm_out = nm;
}

template <typename PointT>
static
PointT CenterPoint(PointCloud<PointT> &cloud)
{
    PointT center;
    center.x = 0;
    center.y = 0;
    center.z = 0;
    for( size_t p=0; p<cloud.size(); p++ )
    {
        center.x += cloud[p].x;
        center.y += cloud[p].y;
        center.z += cloud[p].z;
    }
    center.x /= cloud.size();
    center.y /= cloud.size();
    center.z /= cloud.size();
}

static
float compute_dist(PointXYZ &pt1, PointXYZ &pt2)
{
    return sqrt( (pt1.x-pt2.x)*(pt1.x-pt2.x)+
                 (pt1.y-pt2.y)*(pt1.y-pt2.y)+
                 (pt1.z-pt2.z)*(pt1.z-pt2.z)  );
}

float compute_dist( PointCloud<PointXYZRGB>::Ptr cloud_a, 
                    PointCloud<PointXYZRGB>::Ptr cloud_b  )    
{
    // compare A to B
    pcl::search::KdTree<PointXYZRGB> tree_b;
    tree_b.setInputCloud (cloud_b);
    //float max_dist_a = -std::numeric_limits<float>::max ();
    float dist_a = 0;
    for (size_t i = 0; i < cloud_a->points.size (); ++i)
    {
        std::vector<int> indices (1);
        std::vector<float> sqr_distances (1);
        tree_b.nearestKSearch (cloud_a->points[i], 1, indices, sqr_distances);

        dist_a += sqr_distances[0];
        //if (sqr_distances[0] > max_dist_a)
        //    max_dist_a = sqr_distances[0];
    }

    // compare B to A
    pcl::search::KdTree<PointXYZRGB> tree_a;
    tree_a.setInputCloud (cloud_a);
    //float max_dist_b = -std::numeric_limits<float>::max ();
    float dist_b = 0;
    for (size_t i = 0; i < cloud_b->points.size (); ++i)
    {
        std::vector<int> indices (1);
        std::vector<float> sqr_distances (1);
        tree_a.nearestKSearch (cloud_b->points[i], 1, indices, sqr_distances);

        dist_b += sqr_distances[0];
        //if (sqr_distances[0] > max_dist_b)
        //    max_dist_b = sqr_distances[0];
    }

    //max_dist_a = std::sqrt (max_dist_a);
    //max_dist_b = std::sqrt (max_dist_b);
    //return std::max (max_dist_a, max_dist_b);
    return (dist_a + dist_b) / (cloud_a->points.size()+cloud_b->points.size());
}

template <typename PointT> 
static float compute_dist(const Plane &plane, const PointT &pt)
{
    return abs( plane.coef[0]*pt.x + 
                plane.coef[1]*pt.y + 
                plane.coef[2]*pt.z - plane.coef[3] );
}

float compute_dist(const Plane &pl1, const Plane &pl2)
{
    float a = pl1.coef[0]*pl1.coef[3] - pl2.coef[0]*pl2.coef[3];
    float b = pl1.coef[1]*pl1.coef[3] - pl2.coef[1]*pl2.coef[3];
    float c = pl1.coef[2]*pl1.coef[3] - pl2.coef[2]*pl2.coef[3];    
    return sqrt(a*a + b*b + c*c);    
}

/*
static
bool IsSamePlane(const Plane &plane1, const Plane &plane2 )
{
    #define COS_30 (0.86602540378443860)
    
    // angle
    double dotprod = plane1.coef[0]*plane2.coef[0] + 
                     plane1.coef[1]*plane2.coef[1] + 
                     plane1.coef[2]*plane2.coef[2];
    if( abs(dotprod) < COS_30 ) return false;

    if( dotprod > 0)
    {
        return abs(plane1.coef[3] - plane2.coef[3]) < 0.05;
    }
    else
    {
        return abs(plane1.coef[3] + plane2.coef[3]) < 0.05;
    }    
}
*/

template<typename PointT>
static
bool IsSamePlane(const Plane &plane1, const PointT &pt1,
                 const Plane &plane2, const PointT &pt2 )
{   
    // angle
    double dotprod = plane1.coef[0]*plane2.coef[0] + 
                     plane1.coef[1]*plane2.coef[1] + 
                     plane1.coef[2]*plane2.coef[2];
    if( abs(dotprod) < COS_30 ) return false;

    PointT vec;
    vec.x = pt1.x-pt2.x; vec.y = pt1.y-pt2.y; vec.z = pt1.z-pt2.z;
    float norm = sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
    vec.x /= norm; vec.y /= norm; vec.z /= norm;

    if( abs(plane1.coef[0]*vec.x + 
            plane1.coef[1]*vec.y + 
            plane1.coef[2]*vec.z  ) > COS_75 ) return false;

    if( abs(plane2.coef[0]*vec.x + 
            plane2.coef[1]*vec.y + 
            plane2.coef[2]*vec.z  ) > COS_75 ) return false;

    return true;
}

template <typename P1T, typename P2T>
static bool isSameSide(Plane &plane, P1T p1, P2T p2)
{
    float dist_1
     = plane.coef[0] * p1.x + 
       plane.coef[1] * p1.y + 
       plane.coef[2] * p1.z -
       plane.coef[3];

    float dist_2
     = plane.coef[0] * p2.x + 
       plane.coef[1] * p2.y + 
       plane.coef[2] * p2.z -
       plane.coef[3];

    return dist_1 * dist_2 > 0;
}

template <typename PointT>
static bool IsInside(Plane &plane1, Plane &plane2, PointT &pt)
{
    double d1 = plane1.coef[0] * pt.x + 
                plane1.coef[1] * pt.y + 
                plane1.coef[2] * pt.z - plane1.coef[3];
    double d2 = plane2.coef[0] * pt.x + 
                plane2.coef[1] * pt.y + 
                plane2.coef[2] * pt.z - plane2.coef[3];;

    return d1*d2 < 0;
}

static
void ThreadQuickShift( Mat* image, Mat* image_seg )
{
    Segmentation_QuickShift(*image, *image_seg, 3, 10);    
}

static
void ThreadSiftTrack( ImageTracker* _imgTracker, Mat* image )
{
    _imgTracker->AddImage(*image);    
}

static
void ThreadSegment( SegLCCP2DSeg* _cloudSeg, 
                    Mat* image, PointCloud<PointXYZRGB>::Ptr cloud,
                    string* param, 
                    vector<float>* camera_K, vector<float>* camera_RT, 
                    rl_msgs::SegmentationScene* segscene            )
{
    sensor_msgs::Image msg_img;
    cv_bridge::CvImage cv_img;

    cv_img.image = *image;
    cv_img.encoding = sensor_msgs::image_encodings::BGR8;
    cv_img.toImageMsg(msg_img);

    _cloudSeg->Segment( segscene, msg_img,cloud,*param,*camera_K, *camera_RT );
}

bool VisModelBuilder::IsSameFace( VisFace &f1, VisFace &f2 )
{
    return IsSamePlane( f1.plane, f1.cloud_prj->points[0], 
                        f2.plane, f2.cloud_prj->points[0] );
}

void VisModelBuilder::transform( VisFace &face_in, VisFace &face_out, 
                                 const CloudTracker::Matrix4 &tf )
{
    transformPointCloud(*face_in.cloud,     *face_out.cloud,     tf);
    transformPointCloud(*face_in.cloud_prj, *face_out.cloud_prj, tf);
    rotatePlane( face_in.plane, face_out.plane, tf );

    face_out.coefficients->values[0] =  face_out.plane.coef[0];
    face_out.coefficients->values[1] =  face_out.plane.coef[1];
    face_out.coefficients->values[2] =  face_out.plane.coef[2];
    face_out.coefficients->values[3] = -face_out.plane.coef[3];
}

void VisModelBuilder::transform( 
    VisConvexHull &vchull_in, VisConvexHull &vchull_out, 
    const CloudTracker::Matrix4 &tf )
{
    transformPointCloud(*vchull_in.cloud,      *vchull_out.cloud,      tf);
    transformPointCloud(*vchull_in.cloud_hull, *vchull_out.cloud_hull, tf);

    PointCloud<PointXYZ> cloud;
    fromPCLPointCloud2(vchull_in.polymesh.cloud, cloud);
    transformPointCloud(cloud, cloud, tf);
    toPCLPointCloud2(cloud, vchull_out.polymesh.cloud); 

    PointXYZ pt, pt_tf;
    pt.x = vchull_in.cx; pt.y = vchull_in.cy; pt.z = vchull_in.cz;
    ::transform(pt, pt_tf, tf);
    vchull_out.cx = pt_tf.x; vchull_out.cy = pt_tf.y; vchull_out.cz = pt_tf.z;
}

void VisModelBuilder::AddBGPlane(const Plane &plane)
{
    PointXYZ pt(0,0,0);
    bool found = false;
    for( size_t bg=0; bg<_planes_bg.size(); bg++ )
    {   
        if( IsSamePlane(plane,pt,_planes_bg[bg],pt) )
        {
            found = true;
            break;
        }
    }
    if( !found )
    {
        cout << "Add Background: [" << plane.coef[0] << ","
                                    << plane.coef[1] << ","
                                    << plane.coef[2] << ","
                                    << plane.coef[3] << "]" << endl;
        Plane plane_cam;
        rotatePlane(plane, plane_cam, _tf_world2cam);
        _planes_bg.push_back(plane);
        _planes_bg_cam.push_back(plane_cam);
    }
}

void VisModelBuilder::AddGroundPlane(const Plane &plane)
{
    _plane_ground = plane;
    AddBGPlane(plane);
}

void VisModelBuilder::AddStaticBoxOpened(const vector<float> &box)
{
    shape_msgs::Plane back;
    back.coef[0] =  -1; back.coef[1] =  0; back.coef[2] =  0;
    back.coef[3] =  -box[1];
    AddBGPlane(back);

    shape_msgs::Plane right;
    right.coef[0] =  0; right.coef[1] = 1; right.coef[2] =  0;
    right.coef[3] =  box[2];
    AddBGPlane(right);

    shape_msgs::Plane left;
    left.coef[0] =  0; left.coef[1] = -1; left.coef[2] =  0;
    left.coef[3] = -box[3];
    AddBGPlane(left);

    shape_msgs::Plane bottom;
    bottom.coef[0] =  0; bottom.coef[1] = 0; bottom.coef[2] = 1;
    bottom.coef[3] = box[4];
    AddBGPlane(bottom);

    stringstream ss;
    ss << "workspace=";
    for( int i=0; i<5; i++ ) ss << box[i] << " ";
    ss << "INF";
    if( _param.compare("")!=0 ) _param += ",";
    _param += ss.str();
}

void VisModelBuilder::AddBGPlane(rl_msgs::SegmentationScene &segscene)
{
    // add plane
    for( size_t bg=0; bg<segscene.planes_bg.size(); bg++ )
    {
        Plane plane_world;
        rotatePlane(segscene.planes_bg[bg], plane_world, _tf_cam2world);
        AddBGPlane(plane_world);
    }
}

void VisModelBuilder::AddVisModel( rl_msgs::SegmentationScene &segscene )
{       
    // add models    
    PointCloud<PointXYZRGB>::Ptr cloud_all_cam(new PointCloud<PointXYZRGB>);
    for( size_t o=0; o<segscene.objects.size(); o++ )
    {
        VisModel model;

        _seg2model.insert(pair<size_t,size_t>(o,o));

        PointCloud<PointXYZRGB>::Ptr cloud_cam(new PointCloud<PointXYZRGB>);
        fromRLMsg(segscene.objects[o], cloud_cam);
        *cloud_all_cam += *cloud_cam;

        PointCloud<PointXYZRGB>::Ptr cloud_tran(new PointCloud<PointXYZRGB>);
        transformPointCloud(*cloud_cam, *cloud_tran, _tf_cam2world);
        model.cloud = cloud_tran;

/*
        bool small = true;
        for( size_t f=0; f<segscene.objects[o].faces.size(); f++ )
        {
            float area = segscene.objects[o].faces[f].sizes[0] * 
                         segscene.objects[o].faces[f].sizes[1];
            if( area > 0.03 * 0.03 )
            {
                small = false;
                break;
            }
        }
        if( small ) continue;
*/
        UpdateFaces(segscene.objects[o].faces, model.faces);
        
        model.tf << 1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;
        model.tobeUpdated = true;        
        _models.push_back(model);
    }
    
    UpdateUncertainVoxels(cloud_all_cam);    
}

void VisModelBuilder::UpdateVisModel( rl_msgs::SegmentationScene &segscene,
                                      multimap<size_t,size_t> &end2beg,
                                      vector<CloudTracker::Matrix4> &TFs    )
{
#if 0
    AddVisModel(segscene, _models_new); // TODO dirty code
    for( multimap<size_t,size_t>::iterator 
         it_end2beg  = end2beg.begin(); 
         it_end2beg != end2beg.end();   it_end2beg++ )
    {
        size_t e = it_end2beg->first;
        size_t b = it_end2beg->second;        
        size_t o = _seg2model.find(b)->second;
        seg2model.insert(pair<size_t,size_t>(e,o));

        _models_new[e].prev = &_models[o];

        CloudTracker::Matrix4 tf = _tf_cam2world*TFs[b]*_tf_world2cam;
        _models_new[o].tf = tf * _models[o].tf;
    }
#else    
    PointCloud<PointXYZRGB>::Ptr cloud_all_cam(new PointCloud<PointXYZRGB>);
    for( size_t o=0; o<_models.size(); o++ )
    {
        _models[o].tobeUpdated = false;
    }    

    map<size_t,size_t> seg2model;        
    for( multimap<size_t,size_t>::iterator 
         it_end2beg  = end2beg.begin(); 
         it_end2beg != end2beg.end();   it_end2beg++ )
    {
        size_t e = it_end2beg->first;
        size_t b = it_end2beg->second;        
        size_t o = _seg2model.find(b)->second;
        seg2model.insert(pair<size_t,size_t>(e,o));

        if( _models[o].tobeUpdated==false )
        {
            // Add new cloud
            PointCloud<PointXYZRGB>::Ptr cloud_cam(new PointCloud<PointXYZRGB>);
            fromRLMsg(segscene.objects[e], cloud_cam);

            // temporary code // TODO fix
            if( cloud_cam->size() > _models[o].cloud->size()*2 ) continue;

            *cloud_all_cam += *cloud_cam;

            // Transform prev -> cur pose
            CloudTracker::Matrix4 tf = _tf_cam2world*TFs[b]*_tf_world2cam;
            _models[o].tf = tf * _models[o].tf;

            transformPointCloud(*_models[o].cloud, *_models[o].cloud, tf);
            
            PointCloud<PointXYZRGB>::Ptr cloud_new(new PointCloud<PointXYZRGB>);            
            transformPointCloud(*cloud_cam, *cloud_new, _tf_cam2world);

            *_models[o].cloud += *cloud_new;

cout << "before: " << _models[o].cloud->size() << endl;
VoxelGrid<PointXYZRGB> sor;
sor.setInputCloud (_models[o].cloud);
sor.setLeafSize (_resolution_uncertain, _resolution_uncertain, _resolution_uncertain);
sor.filter (*_models[o].cloud);
cout << "after: " << _models[o].cloud->size() << endl;

            UpdateFaces(segscene.objects[e].faces, _models[o].faces, tf);

            _models[o].tobeUpdated = true;
        }
    }
    _seg2model = seg2model;

    // TODO 
    // new object
    
    UpdateUncertainVoxels(cloud_all_cam);    
#endif
}

void VisModelBuilder::GenerateFace( PointCloud<PointXYZRGB>::Ptr cloud, 
                                    VisFace &face )
{
    face.cloud = cloud;
    face.coefficients = ModelCoefficients::Ptr(new ModelCoefficients);
    face.inliers = PointIndices::Ptr(new PointIndices);

    SACSegmentation<PointXYZRGB> seg;    
    seg.setOptimizeCoefficients (true);
    seg.setModelType (SACMODEL_PLANE);
    seg.setMethodType (SAC_RANSAC);
    seg.setDistanceThreshold (0.01);
    seg.setInputCloud (face.cloud);
    seg.segment (*face.inliers, *face.coefficients);

    // Correct Plane Normal
    Normal normal_tran;
    Normal normal( face.coefficients->values[0],
                   face.coefficients->values[1],
                   face.coefficients->values[2]  );
    ::transform(normal, normal_tran, _tf_world2cam);

    if( normal_tran.normal_z > COS_75 )
    {
        face.coefficients->values[0] = -face.coefficients->values[0];
        face.coefficients->values[1] = -face.coefficients->values[1];
        face.coefficients->values[2] = -face.coefficients->values[2];
        face.coefficients->values[3] = -face.coefficients->values[3];
    }
    face.plane.coef[0] = face.coefficients->values[0];
    face.plane.coef[1] = face.coefficients->values[1];
    face.plane.coef[2] = face.coefficients->values[2];
    face.plane.coef[3] =-face.coefficients->values[3];

    // Project the model inliers
    face.cloud_prj = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
    ProjectInliers<PointXYZRGB> proj;
    proj.setModelType (SACMODEL_PLANE);
    proj.setInputCloud (face.cloud);
    if (face.inliers->indices.size() >= 3 ) proj.setIndices (face.inliers);
    proj.setModelCoefficients (face.coefficients);
    proj.filter (*face.cloud_prj);

    if( face.cloud_prj->size() < 5 )
    {
        face.cloud_prj->push_back(face.cloud_prj->points[0]);
    }

    // Find the extreme points
    face.cloud_hull = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
    ConvexHull<PointXYZRGB> chull;
    chull.setComputeAreaVolume(true);
    chull.setDimension(2);
    chull.setInputCloud (face.cloud_prj);
    chull.reconstruct(face.polymesh);
    face.area = chull.getTotalArea();

    fromPCLPointCloud2(face.polymesh.cloud, *face.cloud_hull);
}

void VisModelBuilder::UpdateFaces(vector<rl_msgs::SegmentationFace> &rl_faces,
                                  vector<VisFace> &faces,
                                  const CloudTracker::Matrix4 &tf             )
{
    // transform cloud
    for( size_t f=0; f<faces.size(); f++ )
    {
        transform(faces[f], faces[f], tf);            
    }

    for( size_t f=0; f<rl_faces.size(); f++ )
    {   
        VisFace face;

        PointCloud<PointXYZRGB>::Ptr cloud_face(new PointCloud<PointXYZRGB>);
        fromRLMsg(rl_faces[f], cloud_face);

        PointCloud<PointXYZRGB>::Ptr cloud_face_tran(new PointCloud<PointXYZRGB>);
        transformPointCloud(*cloud_face, *cloud_face_tran, _tf_cam2world);
                
        GenerateFace(cloud_face_tran, face);
        
        bool found = false;
        float dist_min;
        size_t f2 = 0;        
        for( f2=0; f2<faces.size(); f2++ )
        {
            if( IsSameFace(face,faces[f2]) )
            {
                found = true;   
                break;             
            }
        }

        if( found )
        {
            *faces[f2].cloud += *face.cloud; //TODO update wisely
cout << "before: " << faces[f2].cloud->size() << endl;
VoxelGrid<PointXYZRGB> sor;
sor.setInputCloud (faces[f2].cloud);
sor.setLeafSize (_resolution_uncertain, _resolution_uncertain, _resolution_uncertain);
sor.filter (*faces[f2].cloud);
cout << "after: " << faces[f2].cloud->size() << endl;
            SACSegmentation<PointXYZRGB> seg;    
            seg.setOptimizeCoefficients (true);
            seg.setModelType (SACMODEL_PLANE);
            seg.setMethodType (SAC_RANSAC);
            seg.setDistanceThreshold (0.01);
            seg.setInputCloud (faces[f2].cloud);
            seg.segment (*faces[f2].inliers, *faces[f2].coefficients);
            faces[f2].plane.coef[0] = faces[f2].coefficients->values[0];
            faces[f2].plane.coef[1] = faces[f2].coefficients->values[1];
            faces[f2].plane.coef[2] = faces[f2].coefficients->values[2];
            faces[f2].plane.coef[3] =-faces[f2].coefficients->values[3];

            ProjectInliers<PointXYZRGB> proj;
            proj.setModelType (SACMODEL_PLANE);
            proj.setInputCloud (faces[f2].cloud);
            proj.setIndices (faces[f2].inliers);
            proj.setModelCoefficients (faces[f2].coefficients);
            proj.filter (*faces[f2].cloud_prj);

            ConvexHull<PointXYZRGB> chull;
            chull.setInputCloud (faces[f2].cloud_prj);  
            chull.reconstruct (*faces[f2].cloud_hull);
        }
        else
        {
            faces.push_back(face);
        }
    }
}

void VisModelBuilder::GenerateUncertainVoxels(
    PointCloud<PointXYZRGB>::Ptr cloud_cam,
    PointCloud<PointXYZUncertain> &cloud_uncertain
)
{
    const double resolution = _resolution_uncertain;
    const double len_vec = resolution;

    // extend points to downward
    PointCloud<PointXYZRGB>::Ptr cloud_tmp(new PointCloud<PointXYZRGB>);
    copyPointCloud(*cloud_cam,*cloud_tmp);

    // Generate Voxels
    PointCloud<PointXYZUncertain> cloud_uncertain_cam;
    
    PointCloud<PointXYZRGB>::Ptr cloud_grid (new PointCloud<PointXYZRGB>);
    VoxelGrid<PointXYZRGB> sor;
    sor.setInputCloud (cloud_tmp);
    sor.setLeafSize (resolution, resolution, resolution);
    sor.filter (*cloud_grid);
    
    PointXYZ pt_cam(0,0,0);
    for( size_t p=0; p<cloud_grid->size(); p++ )
    {
        float certain = false;
        PointXYZRGB pt = (*cloud_grid)[p];

        double x_vec=pt.x, y_vec=pt.y, z_vec=pt.z; 
        double norm = sqrt(x_vec*x_vec + y_vec*y_vec + z_vec*z_vec);
        x_vec /= norm;    y_vec /= norm;    z_vec /= norm;
        x_vec *= len_vec; y_vec *= len_vec; z_vec *= len_vec;

        pt.x = pt.x + x_vec;
        pt.y = pt.y + y_vec;
        pt.z = pt.z + z_vec;
        bool reach = false;
        while( !reach )
        {                
            for( size_t pl=0; pl<_planes_bg_cam.size(); pl++ )
            {                    
                Plane &plane = _planes_bg_cam[pl];
                if( !isSameSide(plane, pt, pt_cam) )
                {
                    reach = true;
                    break;
                }
            }
            if( reach ) break;

            PointXYZUncertain pt_uncertain;
            pt_uncertain.x = pt.x;
            pt_uncertain.y = pt.y;
            pt_uncertain.z = pt.z;
            for( size_t o=0; o<32; o++ )
            {
                pt_uncertain.type[o] = HIDDEN;
                pt_uncertain.gravity[o] = false;
                pt_uncertain.n_face_prjs[o] = 0;
                pt_uncertain.idxOwner = -1;
            }            

            cloud_uncertain_cam.push_back(pt_uncertain);

            pt.x = pt.x + x_vec;
            pt.y = pt.y + y_vec;
            pt.z = pt.z + z_vec;
        }
    }
    
    transformPointCloud(cloud_uncertain_cam,cloud_uncertain,_tf_cam2world);    
}

void VisModelBuilder::UpdateUncertainVoxels(
  PointCloud<PointXYZRGB>::Ptr cloud_all_cam )
{
    const double resolution_sq = _resolution_uncertain*_resolution_uncertain;

    PointCloud<PointXYZUncertain> cloud_uncertain_old;
    copyPointCloud(*_cloud_uncertain, cloud_uncertain_old);
    
    _cloud_uncertain->clear();    
    GenerateUncertainVoxels(cloud_all_cam, *_cloud_uncertain);
    _kdtree_uncertain.setInputCloud(_cloud_uncertain);

    // Prepare the points cloud in the camera coordinates
    PointCloud<PointXY>::Ptr cloud_xy(new PointCloud<PointXY>);        
    for( size_t p=0; p<cloud_all_cam->size(); p++ )
    {
        PointXY pt;                 
        Projection_K( (*cloud_all_cam)[p], _camera_K, &pt.x, &pt.y);
        cloud_xy->push_back(pt);
    }
    KdTreeFLANN<PointXY> kdtree_cam;
    kdtree_cam.setInputCloud (cloud_xy);

    // Prepare for the gravity
    PointCloud<PointXYZRGB>::Ptr cloud_uncertain_xyz(new PointCloud<PointXYZRGB>);
    for( size_t p=0; p<_cloud_uncertain->size(); p++ )
    {
        PointXYZRGB pt;
        pt.x = (*_cloud_uncertain)[p].x;
        pt.y = (*_cloud_uncertain)[p].y;
        pt.z = (*_cloud_uncertain)[p].z;
        cloud_uncertain_xyz->push_back(pt);
    }

    // Camera Ray    
    for( size_t o=0; o<_models.size(); o++ )
    {
        VisModel &model = _models[o];
        if( model.tobeUpdated==false ) continue;

#if 0 // no make SENSE!!!!        
        // cut off the newly observed volume
        PointCloud<PointXYZUncertain> cloud_uncertain_old_tran;
        transformPointCloud( cloud_uncertain_old, 
                             cloud_uncertain_old_tran, model.tf );
        transformPointCloud( cloud_uncertain_old_tran, 
                             cloud_uncertain_old_tran, _tf_world2cam );

        for( size_t p=0; p<cloud_uncertain_old_tran.size(); p++ )
        {
            PointXY pt;                 
            Projection_K( (*cloud_all_cam)[p], _camera_K, &pt.x, &pt.y);

            vector<int> idxes;
            vector<float> dists;            
            kdtree_cam.nearestKSearch(pt, 1, idxes, dists);
            if( dists[0] > 0.005f*0.005f )
            {
                (*_cloud_uncertain)[idxes[0]].type[o] = EMPTY;
            }
        }
#endif
        // Gravity
        ModelCoefficients::Ptr coefficients_z (new ModelCoefficients());
        coefficients_z->values.resize (4);
        coefficients_z->values[0] = 0;
        coefficients_z->values[1] = 0;
        coefficients_z->values[2] = 1;
        coefficients_z->values[3] = 0;

        PointCloud<PointXYZRGB>::Ptr cloud_uncertain_zrj(new PointCloud<PointXYZRGB>);
        ProjectInliers<PointXYZRGB> proj_zrj;
        proj_zrj.setModelType(SACMODEL_PLANE);
        proj_zrj.setModelCoefficients(coefficients_z);
        proj_zrj.setInputCloud(cloud_uncertain_xyz);
        proj_zrj.filter(*cloud_uncertain_zrj);

        PointCloud<PointXYZRGB>::Ptr cloud_zrj(new PointCloud<PointXYZRGB>);
        proj_zrj.setInputCloud(model.cloud);
        proj_zrj.filter(*cloud_zrj);

        KdTreeFLANN<PointXYZRGB> kdtree_zrj;
        kdtree_zrj.setInputCloud (cloud_zrj);

        for( size_t p=0; p<_cloud_uncertain->size(); p++ )
        {
            (*_cloud_uncertain)[p].gravity[o] = false;
            if( (*_cloud_uncertain)[p].type[o]==EMPTY ) continue;

            PointXYZRGB &pt_zrj = (*cloud_uncertain_zrj)[p];
            
            vector<int> idxes;
            vector<float> dists;            
            kdtree_zrj.nearestKSearch(pt_zrj, 1, idxes, dists);
            if( dists[0] <= resolution_sq &&
                model.cloud->points[idxes[0]].z > (*_cloud_uncertain)[p].z )
            {
                (*_cloud_uncertain)[p].gravity[o] = true;
            }
        }

        // Face Ray
        for( size_t p=0; p<_cloud_uncertain->size(); p++ )
        {            
            (*_cloud_uncertain)[p].n_face_prjs[o] = 0;            
        }
        for( size_t f=0; f<model.faces.size(); f++ )
        {
            VisFace &face = model.faces[f];
            face.idxes_uncertain.clear();

            PointCloud<PointXYZRGB>::Ptr cloud_frj(new PointCloud<PointXYZRGB>);
            ProjectInliers<PointXYZRGB> proj_frj;
            proj_frj.setModelType(SACMODEL_PLANE);
            proj_frj.setInputCloud(cloud_uncertain_xyz);    
            proj_frj.setModelCoefficients(face.coefficients);
            proj_frj.filter(*cloud_frj);

            KdTreeFLANN<PointXYZRGB> kdtree_face;
            kdtree_face.setInputCloud (face.cloud_prj);

            for( size_t p=0; p<_cloud_uncertain->size(); p++ )
            {
                if( (*_cloud_uncertain)[p].type[o] == EMPTY ) continue;

                PointXYZRGB pt;
                pt.x = (*cloud_frj)[p].x; 
                pt.y = (*cloud_frj)[p].y; 
                pt.z = (*cloud_frj)[p].z;
                vector<int> idxes;
                vector<float> dists;
                kdtree_face.nearestKSearch(pt, 1, idxes, dists);

                if( dists[0] < 0.005f*0.005f )
                {                    
                    (*_cloud_uncertain)[p].n_face_prjs[o] += 1;
                    face.idxes_uncertain.push_back(p);
                }
            }
        }
    }
}

void VisModelBuilder::SetCameraRT(const vector<float> &camera_RT)
{    
    copyTo(camera_RT, _tf_cam2world);
    _tf_world2cam = _tf_cam2world.inverse();

    _camera_RT=camera_RT;
    _cameras_pos.push_back(PointXYZ(_camera_RT[3],_camera_RT[7],_camera_RT[11]));
}

void VisModelBuilder::Update( Mat &image, Mat &depth,
                              std::vector<float> camera_RT )
{
    if( camera_RT.size()==0 ) camera_RT = _camera_RT;

    mtx_update_2d.lock();

    Mat image_seg;
    thread t_seg(ThreadQuickShift, &image, &image_seg);
    thread t_sift(ThreadSiftTrack, _imgTracker, &image);
    _depths.push_back(depth);    
    t_seg.join();
    _images_seg.push_back(image_seg);
    t_sift.join();

    mtx_update_2d.unlock();
}

static float compute_vis_likelihood(float x)
{
    return 1/(1+exp(-4*(x-1)));
}

static float compute_vis_penalty(float x)
{
    return 1/(1+exp(-8*(x-2)));
}

template<typename PointT>
static double compute_variance(PointCloud<PointT> &cloud)
{
    double cx=0, cy=0, cz=0;
    for( size_t p=0; p<cloud.size(); p++ )
    {
        cx += cloud[p].x;
        cy += cloud[p].y;
        cz += cloud[p].z;
    }
    cx /= cloud.size();
    cy /= cloud.size();
    cz /= cloud.size();

    double var = 0;
    for( size_t p=0; p<cloud.size(); p++ )
    {
        var += (cloud[p].x - cx)*(cloud[p].x - cx) + 
               (cloud[p].y - cy)*(cloud[p].y - cy) + 
               (cloud[p].z - cz)*(cloud[p].z - cz);
    }
    var /= cloud.size();

    return var;
}

template<typename PointT>
static 
void GenerateVischull( PointCloud<PointT> &cloud_in, 
                       VisModelBuilder::VisConvexHull &vischull               )
{    
    PointCloud<PointXYZRGB>::Ptr cloud (new PointCloud<PointXYZRGB>);
    copyPointCloud(cloud_in, *cloud);

    PointCloud<PointXYZRGB>::Ptr cloud_hull (new PointCloud<PointXYZRGB>);
    ConvexHull<PointXYZRGB> chull;
    chull.setInputCloud (cloud);
    chull.reconstruct (vischull.polymesh);
    chull.reconstruct (*cloud_hull);
    //fromPCLPointCloud2(vischull.polymesh.cloud, *cloud_hull);

    PassThrough<PointXYZRGB> pass;
    pass.setInputCloud (cloud_hull);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (0.1, 2);
    pass.filter (*cloud_hull);
        
    // Center points
    PointT pt_min, pt_max;
    getMinMax3D(*cloud_hull, pt_min, pt_max);
    double cx = (pt_max.x+pt_min.x)/2, 
           cy = (pt_max.y+pt_min.y)/2, 
           cz = (pt_max.z+pt_min.z)/2;
    for( size_t p=0; p<cloud->size(); p++ )
    {
        (*cloud)[p].x -= cx;
        (*cloud)[p].y -= cy;
        (*cloud)[p].z -= cz;        
    }
    for( size_t p=0; p<cloud_hull->size(); p++ )
    {
        (*cloud_hull)[p].x -= cx;
        (*cloud_hull)[p].y -= cy;
        (*cloud_hull)[p].z -= cz;
    }

    vischull.cloud      = cloud;
    vischull.cloud_hull = cloud_hull;
    vischull.cx = cx;
    vischull.cy = cy;
    vischull.cz = cz;
    //vischull.tf_cur2init = model.tf.inverse();
}

bool VisModelBuilder::GenerateGravityModel(size_t o, VisConvexHull &vischull)
{    
    VisModel &model = _models[o];

    PointCloud<PointXYZRGB>::Ptr cloud (new PointCloud<PointXYZRGB>);   
    /*
    if( model.faces.size()==1 )
    {
        VisFace &face = model.faces[0];
        float vec_x = face.plane.coef[0];
        float vec_y = face.plane.coef[1];
        float norm = sqrt(vec_x*vec_x + vec_y*vec_y);
        vec_x /= norm;
        vec_y /= norm;
        if( vec_x<0 )
        {
            vec_x = -vec_x;
            vec_y = -vec_y; 
        }

        for( size_t p=0; p<model.cloud->size(); p++ )
        {
            PointXYZRGB pt;
            pt.x = model.cloud->points[p].x + vec_x * 0.01;
            pt.y = model.cloud->points[p].y + vec_y * 0.01;
            pt.z = model.cloud->points[p].z;
            pt.r = 0;
            pt.g = 0;
            pt.b = 255;
            cloud->push_back(pt);
        }
    }
    else
    */
    {
        ModelCoefficients::Ptr coefficients_x (new ModelCoefficients());
        coefficients_x->values.resize (4);
        coefficients_x->values[0] = 1;
        coefficients_x->values[1] = 0;
        coefficients_x->values[2] = 0;
        coefficients_x->values[3] = 0;

        ModelCoefficients::Ptr coefficients_y (new ModelCoefficients());
        coefficients_y->values.resize (4);
        coefficients_y->values[0] = 0;
        coefficients_y->values[1] = 1;
        coefficients_y->values[2] = 0;
        coefficients_y->values[3] = 0;

        ModelCoefficients::Ptr coefficients_z (new ModelCoefficients());
        coefficients_z->values.resize (4);
        coefficients_z->values[0] = 0;
        coefficients_z->values[1] = 0;
        coefficients_z->values[2] = 1;
        coefficients_z->values[3] = 0;


        PointCloud<PointXYZRGB>::Ptr cloud_uncertain_xyz(new PointCloud<PointXYZRGB>);
        for( size_t p=0; p<_cloud_uncertain->size(); p++ )
        {
            PointXYZRGB pt;
            pt.x = (*_cloud_uncertain)[p].x;
            pt.y = (*_cloud_uncertain)[p].y;
            pt.z = (*_cloud_uncertain)[p].z;
            cloud_uncertain_xyz->push_back(pt);
        }

        PointCloud<PointXYZRGB>::Ptr cloud_uncertain_xrj(new PointCloud<PointXYZRGB>);
        ProjectInliers<PointXYZRGB> proj_xrj;
        proj_xrj.setModelType(SACMODEL_PLANE);
        proj_xrj.setModelCoefficients(coefficients_x);
        proj_xrj.setInputCloud(cloud_uncertain_xyz);
        proj_xrj.filter(*cloud_uncertain_xrj);

        PointCloud<PointXYZRGB>::Ptr cloud_uncertain_yrj(new PointCloud<PointXYZRGB>);
        ProjectInliers<PointXYZRGB> proj_yrj;
        proj_yrj.setModelType(SACMODEL_PLANE);
        proj_yrj.setModelCoefficients(coefficients_y);
        proj_yrj.setInputCloud(cloud_uncertain_xyz);
        proj_yrj.filter(*cloud_uncertain_yrj);

        PointCloud<PointXYZRGB>::Ptr cloud_uncertain_zrj(new PointCloud<PointXYZRGB>);
        ProjectInliers<PointXYZRGB> proj_zrj;
        proj_zrj.setModelType(SACMODEL_PLANE);
        proj_zrj.setModelCoefficients(coefficients_z);
        proj_zrj.setInputCloud(cloud_uncertain_xyz);
        proj_zrj.filter(*cloud_uncertain_zrj);

        PointCloud<PointXYZRGB>::Ptr cloud_xrj(new PointCloud<PointXYZRGB>);
        proj_xrj.setInputCloud(model.cloud);
        proj_xrj.filter(*cloud_xrj);

        PointCloud<PointXYZRGB>::Ptr cloud_yrj(new PointCloud<PointXYZRGB>);
        proj_yrj.setInputCloud(model.cloud);
        proj_yrj.filter(*cloud_yrj);

        PointCloud<PointXYZRGB>::Ptr cloud_zrj(new PointCloud<PointXYZRGB>);
        proj_zrj.setInputCloud(model.cloud);
        proj_zrj.filter(*cloud_zrj);

        KdTreeFLANN<PointXYZRGB> kdtree_xrj;
        kdtree_xrj.setInputCloud (cloud_xrj);
        KdTreeFLANN<PointXYZRGB> kdtree_yrj;
        kdtree_yrj.setInputCloud (cloud_yrj);
        KdTreeFLANN<PointXYZRGB> kdtree_zrj;
        kdtree_zrj.setInputCloud (cloud_zrj);

        float resolution_sq = _resolution_uncertain*_resolution_uncertain;

        for( size_t p=0; p<_cloud_uncertain->size(); p++ )
        {
            if( (*_cloud_uncertain)[p].type[o]==EMPTY ) continue;
            
            int count = 0;

            vector<int> idxes;
            vector<float> dists;            

            PointXYZRGB &pt_xrj = (*cloud_uncertain_xrj)[p];
            kdtree_xrj.nearestKSearch(pt_xrj, 1, idxes, dists);
            if( dists[0] <= resolution_sq ) count++;

            PointXYZRGB &pt_yrj = (*cloud_uncertain_yrj)[p];
            kdtree_yrj.nearestKSearch(pt_yrj, 1, idxes, dists);
            if( dists[0] <= resolution_sq ) count++;

            PointXYZRGB &pt_zrj = (*cloud_uncertain_zrj)[p];
            kdtree_zrj.nearestKSearch(pt_zrj, 1, idxes, dists);
            if( dists[0] <= resolution_sq ) count++;
            
            if( count >= 2)
            {
                PointXYZRGB pt;
                pt.x = (*_cloud_uncertain)[p].x;
                pt.y = (*_cloud_uncertain)[p].y;
                pt.z = (*_cloud_uncertain)[p].z;
                pt.r = 0;
                pt.g = 0;
                pt.b = 0;
                cloud->push_back(pt);
            }
        }
    }


    GenerateVischull(*cloud, vischull);
    vischull.tf_cur2init = model.tf.inverse();

#if 0
    VisModel &model = _models[o];
    PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);

    float vis_likelihood = 0;
    *cloud += *model.cloud;
    for( size_t p=0; p < _cloud_uncertain->size(); p++ )
    {
        PointXYZUncertain &pt = _cloud_uncertain->points[p];
        if( pt.type[o] == EMPTY ) continue;

        if( pt.gravity[o] == true )
        {
            PointXYZRGB pt_add;
            pt_add.x = pt.x; pt_add.y = pt.y; pt_add.z = pt.z;
            pt_add.r = 100; pt_add.g = 100; pt_add.b = 100;
            cloud->push_back(pt_add);

            vis_likelihood += pt.n_face_prjs[o];
        }
    }
    
    GenerateVischull(*cloud, vischull);
    vischull.tf_cur2init = model.tf.inverse();
#endif
}

bool VisModelBuilder::GenerateGravityModel2(size_t o, VisConvexHull &vischull)
{    
    VisModel &model = _models[o];

    PointCloud<PointXYZRGB>::Ptr cloud (new PointCloud<PointXYZRGB>);   
    
    if( model.faces.size()==1 )
    {
        VisFace &face = model.faces[0];
        float vec_x = face.plane.coef[0];
        float vec_y = face.plane.coef[1];
        float norm = sqrt(vec_x*vec_x + vec_y*vec_y);
        vec_x /= norm;
        vec_y /= norm;
        if( vec_x<0 )
        {
            vec_x = -vec_x;
            vec_y = -vec_y; 
        }

        for( size_t p=0; p<model.cloud->size(); p++ )
        {
            PointXYZRGB pt;
            pt.x = model.cloud->points[p].x + vec_x * 0.01;
            pt.y = model.cloud->points[p].y + vec_y * 0.01;
            pt.z = model.cloud->points[p].z;
            pt.r = 0;
            pt.g = 0;
            pt.b = 255;
            cloud->push_back(pt);
        }
    }
    else    
    {
        ModelCoefficients::Ptr coefficients_x (new ModelCoefficients());
        coefficients_x->values.resize (4);
        coefficients_x->values[0] = 1;
        coefficients_x->values[1] = 0;
        coefficients_x->values[2] = 0;
        coefficients_x->values[3] = 0;

        ModelCoefficients::Ptr coefficients_y (new ModelCoefficients());
        coefficients_y->values.resize (4);
        coefficients_y->values[0] = 0;
        coefficients_y->values[1] = 1;
        coefficients_y->values[2] = 0;
        coefficients_y->values[3] = 0;

        ModelCoefficients::Ptr coefficients_z (new ModelCoefficients());
        coefficients_z->values.resize (4);
        coefficients_z->values[0] = 0;
        coefficients_z->values[1] = 0;
        coefficients_z->values[2] = 1;
        coefficients_z->values[3] = 0;


        PointCloud<PointXYZRGB>::Ptr cloud_uncertain_xyz(new PointCloud<PointXYZRGB>);
        for( size_t p=0; p<_cloud_uncertain->size(); p++ )
        {
            PointXYZRGB pt;
            pt.x = (*_cloud_uncertain)[p].x;
            pt.y = (*_cloud_uncertain)[p].y;
            pt.z = (*_cloud_uncertain)[p].z;
            cloud_uncertain_xyz->push_back(pt);
        }

        PointCloud<PointXYZRGB>::Ptr cloud_uncertain_xrj(new PointCloud<PointXYZRGB>);
        ProjectInliers<PointXYZRGB> proj_xrj;
        proj_xrj.setModelType(SACMODEL_PLANE);
        proj_xrj.setModelCoefficients(coefficients_x);
        proj_xrj.setInputCloud(cloud_uncertain_xyz);
        proj_xrj.filter(*cloud_uncertain_xrj);

        PointCloud<PointXYZRGB>::Ptr cloud_uncertain_yrj(new PointCloud<PointXYZRGB>);
        ProjectInliers<PointXYZRGB> proj_yrj;
        proj_yrj.setModelType(SACMODEL_PLANE);
        proj_yrj.setModelCoefficients(coefficients_y);
        proj_yrj.setInputCloud(cloud_uncertain_xyz);
        proj_yrj.filter(*cloud_uncertain_yrj);

        PointCloud<PointXYZRGB>::Ptr cloud_uncertain_zrj(new PointCloud<PointXYZRGB>);
        ProjectInliers<PointXYZRGB> proj_zrj;
        proj_zrj.setModelType(SACMODEL_PLANE);
        proj_zrj.setModelCoefficients(coefficients_z);
        proj_zrj.setInputCloud(cloud_uncertain_xyz);
        proj_zrj.filter(*cloud_uncertain_zrj);

        PointCloud<PointXYZRGB>::Ptr cloud_xrj(new PointCloud<PointXYZRGB>);
        proj_xrj.setInputCloud(model.cloud);
        proj_xrj.filter(*cloud_xrj);

        PointCloud<PointXYZRGB>::Ptr cloud_yrj(new PointCloud<PointXYZRGB>);
        proj_yrj.setInputCloud(model.cloud);
        proj_yrj.filter(*cloud_yrj);

        PointCloud<PointXYZRGB>::Ptr cloud_zrj(new PointCloud<PointXYZRGB>);
        proj_zrj.setInputCloud(model.cloud);
        proj_zrj.filter(*cloud_zrj);

        KdTreeFLANN<PointXYZRGB> kdtree_xrj;
        kdtree_xrj.setInputCloud (cloud_xrj);
        KdTreeFLANN<PointXYZRGB> kdtree_yrj;
        kdtree_yrj.setInputCloud (cloud_yrj);
        KdTreeFLANN<PointXYZRGB> kdtree_zrj;
        kdtree_zrj.setInputCloud (cloud_zrj);

        float resolution_sq = _resolution_uncertain*_resolution_uncertain;

        for( size_t p=0; p<_cloud_uncertain->size(); p++ )
        {
            if( (*_cloud_uncertain)[p].type[o]==EMPTY ) continue;
            
            int count = 0;

            vector<int> idxes;
            vector<float> dists;            

            PointXYZRGB &pt_xrj = (*cloud_uncertain_xrj)[p];
            kdtree_xrj.nearestKSearch(pt_xrj, 1, idxes, dists);
            if( dists[0] <= resolution_sq ) count++;

            PointXYZRGB &pt_yrj = (*cloud_uncertain_yrj)[p];
            kdtree_yrj.nearestKSearch(pt_yrj, 1, idxes, dists);
            if( dists[0] <= resolution_sq ) count++;

            PointXYZRGB &pt_zrj = (*cloud_uncertain_zrj)[p];
            kdtree_zrj.nearestKSearch(pt_zrj, 1, idxes, dists);
            if( dists[0] <= resolution_sq ) count++;
            
            if( count >= 2)
            {
                PointXYZRGB pt;
                pt.x = (*_cloud_uncertain)[p].x;
                pt.y = (*_cloud_uncertain)[p].y;
                pt.z = (*_cloud_uncertain)[p].z;
                pt.r = 0;
                pt.g = 0;
                pt.b = 0;
                cloud->push_back(pt);
            }
        }
    }
    GenerateVischull(*cloud, vischull);
    vischull.tf_cur2init = model.tf.inverse();
}

#define DEBUG_MIRROR 0

bool VisModelBuilder::GetRandomMirrorPoints( 
    uint32_t o,    
    const PointCloud<PointXYZRGB>::Ptr cloud, 
    PointCloud<PointXYZRGB>::Ptr cloud_mid, 
    Plane plane,
    VisFace &face,
    PointCloud<PointXYZUncertain>::Ptr cloud_uncertain,
    search::KdTree<PointXYZRGB> &kdtree_others,    
    int flag, float dist_bound_min, PointCloud<PointXYZRGB>::Ptr cloud_other )
{

    if( plane.coef[0] > COS_75 )
    {
        plane.coef[0] = -plane.coef[0];
        plane.coef[1] = -plane.coef[1];
        plane.coef[2] = -plane.coef[2];
        plane.coef[3] = -plane.coef[3];
    }


#if DEBUG_MIRROR
PointCloud<PointXYZRGB>::Ptr cloud_tmp( new PointCloud<PointXYZRGB>);
#endif

    const double len_vec = _resolution_uncertain;
    const double resolution_relax = _resolution_uncertain*4;
    const double resolution_relax_sq  = resolution_relax*resolution_relax;
    const double resolution_tight_sq = _resolution_uncertain*_resolution_uncertain;

    double x_vec = -plane.coef[0],
           y_vec = -plane.coef[1],
           z_vec = -plane.coef[2];         
    double norm = sqrt(x_vec*x_vec + y_vec*y_vec + z_vec*z_vec);
    x_vec /= norm;    y_vec /= norm;    z_vec /= norm;
    x_vec *= len_vec; y_vec *= len_vec; z_vec *= len_vec;
    
    double dist_bd = INFINITY;    
    PointXYZRGB pt_bd;
    
    if( cloud_mid==NULL || cloud_mid->size() < 10 )
    {
        cloud_mid = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);    
        VoxelGrid<PointXYZRGB> sor;
        sor.setInputCloud (cloud);
        sor.setLeafSize( _resolution_uncertain*4, 
                         _resolution_uncertain*4, 
                         _resolution_uncertain*4 );
        sor.filter (*cloud_mid);

        pcl::StatisticalOutlierRemoval<PointXYZRGB> sor_noise;
        sor_noise.setInputCloud (cloud_mid);
        sor_noise.setMeanK (20);
        sor_noise.setStddevMulThresh (1);
        sor_noise.filter (*cloud_mid);
    }    

    for( size_t p=0; p<cloud_mid->size(); p++ )
    {
        PointXYZRGB &pt_org = cloud_mid->points[p];
        VisModelBuilder::PointXYZUncertain pt;            
        pt.x = pt_org.x + x_vec;
        pt.y = pt_org.y + y_vec;
        pt.z = pt_org.z + z_vec;

        int n_reach = 0;        
        while( n_reach < 4 )
        {
            bool reach = true;

            vector<int> indexes (1);
            vector<float> dists (1);            
            _kdtree_uncertain.nearestKSearch(pt, 1, indexes, dists);
            
            if( dists[0] <= resolution_relax_sq )
            {                    
                PointXYZUncertain &pt_uncertain = (*cloud_uncertain)[indexes[0]];
                if( pt_uncertain.idxOwner == -1 ||
                    pt_uncertain.idxOwner == o     )
                {
                    if( (flag & MASK_FROM) == FROM_ALL )
                    {
                        reach = false;
                    }
                    else if( (flag & MASK_FROM) == FROM_ONEFACE && 
                             pt_uncertain.n_face_prjs[o] > 0)
                    {
                        reach = false;
                    }
                    else if( (flag & MASK_FROM) == FROM_TWOFACES && 
                             pt_uncertain.n_face_prjs[o] > 1)
                    {
                        reach = false;
                    }
                }
            }

#if 0
            PointXYZRGB pt_rgb;
            pt_rgb.x = pt.x; pt_rgb.y = pt.y; pt_rgb.z = pt.z;
            kdtree_others.nearestKSearch(pt_rgb, 1, indexes, dists);
            if( dists[0] <= resolution_tight_sq )
            {
                reach = true;
                n_reach++;
#if DEBUG_MIRROR
PointXYZRGB pt_tmp;
pt_tmp.x = pt.x;
pt_tmp.y = pt.y;
pt_tmp.z = pt.z;
pt_tmp.r = 255;
pt_tmp.g = 0;
pt_tmp.b = 255;
cloud_tmp->push_back(pt_tmp);
#endif
                break; // escape loop immediately
            } 
#endif

            if( reach ) n_reach++;
            else        n_reach=0;

            pt.x = pt.x + x_vec;
            pt.y = pt.y + y_vec;
            pt.z = pt.z + z_vec;

#if DEBUG_MIRROR
if( n_reach < 1 )
{
    PointXYZRGB pt_tmp;
    pt_tmp.x = pt.x;
    pt_tmp.y = pt.y;
    pt_tmp.z = pt.z;
    pt_tmp.r = 0;
    pt_tmp.g = 255;
    pt_tmp.b = 0;
    cloud_tmp->push_back(pt_tmp);
}
#endif
        }        
        pt.x = pt.x - x_vec*(n_reach) + plane.coef[0]*resolution_relax;
        pt.y = pt.y - y_vec*(n_reach) + plane.coef[1]*resolution_relax;
        pt.z = pt.z - z_vec*(n_reach) + plane.coef[2]*resolution_relax;

#if DEBUG_MIRROR
PointXYZRGB pt_tmp;
pt_tmp.x = pt.x;
pt_tmp.y = pt.y;
pt_tmp.z = pt.z;
pt_tmp.r = 255;
pt_tmp.g = 0;
pt_tmp.b = 0;
cloud_tmp->push_back(pt_tmp);
#endif

        float dist_pt = abs(pt.x*plane.coef[0] + 
                            pt.y*plane.coef[1] + 
                            pt.z*plane.coef[2] - plane.coef[3]);

        if( dist_pt < dist_bound_min ) continue;
        if( dist_bd > dist_pt )
        {
            dist_bd = dist_pt;
            pt_bd.x=pt.x; pt_bd.y=pt.y; pt_bd.z=pt.z;
        }
    }

    PointCloud<PointXYZRGB>::Ptr cloud_mirror(new PointCloud<PointXYZRGB>);    
    if( (flag & MASK_SHAPE) == GRAVITY_SHAPE )
    {        
        for( size_t p=0; p<cloud->size(); p++ )
        {
            PointXYZRGB &pt = cloud->points[p];

            PointXYZRGB pt_mirror;
            pt_mirror.x = pt.x;
            pt_mirror.y = pt.y;
            pt_mirror.z = pt_bd.z;
            pt_mirror.r = 0;//cloud->points[p].r;
            pt_mirror.g = 0;//cloud->points[p].g;
            pt_mirror.b = 255;//cloud->points[p].b;

            cloud_mirror->push_back(pt_mirror);        
        }

        GenerateFace(cloud_mirror, face);
    }
    else
    {
        float rnd;
        if(      (flag & MASK_SHAPE) == MINIMUM_SHAPE ) rnd = 0;
        else if( (flag & MASK_SHAPE) == MAXIMUM_SHAPE ) rnd = 1;
        else if( (flag & MASK_SHAPE) == MEDIUM_SHAPE  ) rnd = 0.5;
        else rnd = (float)rand()/(float)RAND_MAX;

        float dist_rnd = dist_bd * rnd;
        if( dist_rnd == INFINITY || dist_rnd != dist_rnd ||
            dist_rnd < dist_bound_min ) dist_rnd=dist_bound_min;    
                
        for( size_t p=0; p<cloud->size(); p++ )
        {        
            PointXYZRGB &pt = cloud->points[p];
            float offset_dist = plane.coef[0]*pt.x + 
                                plane.coef[1]*pt.y + 
                                plane.coef[2]*pt.z - (-dist_rnd/2+plane.coef[3]);

            //float len = (len_vec*(rnd+1)+offset_dist*2);
            float len = offset_dist*2;
            if( len < 0 ) len = 0;

            PointXYZRGB pt_mirror;
            pt_mirror.x = pt.x + -plane.coef[0]*len;
            pt_mirror.y = pt.y + -plane.coef[1]*len;
            pt_mirror.z = pt.z + -plane.coef[2]*len;
            pt_mirror.r = 0;//cloud->points[p].r;
            pt_mirror.g = 0;//cloud->points[p].g;
            pt_mirror.b = 255;//cloud->points[p].b;

            cloud_mirror->push_back(pt_mirror);        
        }

        GenerateFace(cloud_mirror, face);
    }

#if DEBUG_MIRROR
if( o==2 )
{
cout << "flag: " << hex << flag << dec << endl;
visualization::PCLVisualizer viewer; 
int v1, v2;
viewer.createViewPort (0.0, 0.0, 0.5, 1.0, v1);
viewer.createViewPort (0.5, 0.0, 1.0, 1.0, v2);
viewer.setWindowName("debug");
viewer.setSize(600,480);
viewer.setPosition(600,0);
viewer.setCameraPosition(-1,0,0,1,0,0,0,0,1);
viewer.setBackgroundColor (1, 1, 1);
PointCloud<PointXYZRGB>::Ptr cloud_max(new PointCloud<PointXYZRGB>);    
for( size_t p=0; p<cloud->size(); p++ )
{        
    PointXYZRGB &pt = cloud->points[p];
    float offset_dist = plane.coef[0]*pt.x + 
                        plane.coef[1]*pt.y + 
                        plane.coef[2]*pt.z - (-dist_bd/2+plane.coef[3]);

    //float len = (len_vec*(rnd+1)+offset_dist*2);
    float len = offset_dist*2;
    if( len < 0 ) len = 0;

    PointXYZRGB pt_max;
    pt_max.x = pt.x + -plane.coef[0]*len;
    pt_max.y = pt.y + -plane.coef[1]*len;
    pt_max.z = pt.z + -plane.coef[2]*len;
    pt_max.r = 0;//cloud->points[p].r;
    pt_max.g = 0;//cloud->points[p].g;
    pt_max.b = 0;//cloud->points[p].b;

    cloud_max->push_back(pt_max);        
}
PointCloud<PointXYZRGB>::Ptr cloud_uc_tmp(new PointCloud<PointXYZRGB>);
for( size_t p=0; p<cloud_uncertain->size(); p++ )
{
    if( (flag & MASK_FROM) == FROM_TWOFACES && 
        cloud_uncertain->points[p].n_face_prjs[o] <= 1)
    {
        continue;
    }
    if( (flag & MASK_FROM) == FROM_ONEFACE && 
         cloud_uncertain->points[p].n_face_prjs[o] < 1)
    {
        continue;
    }
    PointXYZRGB pt;
    pt.x = cloud_uncertain->points[p].x + 0.001;
    pt.y = cloud_uncertain->points[p].y + 0.001;
    pt.z = cloud_uncertain->points[p].z + 0.001;
    pt.r = 0;
    pt.g = 255;
    pt.b = 0;
    cloud_uc_tmp->push_back(pt);
}
viewer.addPointCloud(cloud_tmp,"cloud_way",v1);
viewer.addPointCloud(cloud,"cloud_org",v1);
viewer.addPointCloud(cloud_mirror,"cloud_mirror",v1);
viewer.addPointCloud(cloud_max,"cloud_max",v2);
viewer.addPointCloud(cloud_uc_tmp,"cloud_uncertain",v2);
if( cloud_other )
{
    viewer.addPointCloud(cloud_other,"cloud_other",v1);
    viewer.addPointCloud(cloud_other,"cloud_other2",v2);
    viewer.addPointCloud(cloud_mirror,"cloud_mirror2",v2);
}
ModelCoefficients coef_mirror;
coef_mirror.values.resize(4);
coef_mirror.values[0] = plane.coef[0];
coef_mirror.values[1] = plane.coef[1];
coef_mirror.values[2] = plane.coef[2];
coef_mirror.values[3] = -(-dist_bd+plane.coef[3]);
viewer.addPlane(coef_mirror, "plane_mirror", v1);

ModelCoefficients coef_ground;
coef_ground.values.resize(4);
coef_ground.values[0] = _plane_ground.coef[0];
coef_ground.values[1] = _plane_ground.coef[1];
coef_ground.values[2] = _plane_ground.coef[2];
coef_ground.values[3] =-_plane_ground.coef[3];
viewer.addPlane(coef_ground, "plane_ground", v1);
viewer.addPlane(coef_ground, "plane_ground2", v2);

viewer.spin();
}
#endif
    return true;
}

static
bool CompareIdxNVal( pair<size_t,double> iv1, 
                     pair<size_t,double> iv2  )
{
    return iv1.second > iv2.second;
}

static size_t PickRandom(vector<double> &prob_normalized)
{
    double val = (double)rand() / RAND_MAX;

    double cum = 0;
    for( size_t i=0; i<prob_normalized.size(); i++ )
    {
        cum += prob_normalized[i];
        if( val <= cum ) return i;        
    }

    cout << "prob_normalized.size(): " << prob_normalized.size() << endl;
    for( size_t i=0; i<prob_normalized.size(); i++ )
    {
        cout << "[" << i << "] " << prob_normalized[i] << " ";
    }
    cout << endl;

    assert(false && "Never Reach Here!! (PickRandom)");
}

static void Split(vector<double> probs, int count, vector<int> &res )
{
    res.resize(probs.size());

    if( count == 1 )
    {
        size_t rnd = PickRandom(probs);
        for( size_t p=0; p<probs.size(); p++ )
        {
            if( rnd==p ) res[p] = 1;
            else         res[p] = 0;
        }
        return;
    }

    vector<pair<size_t,double> > idxNprobs(probs.size());
    for( size_t i=0; i<probs.size(); i++ )
    {
        idxNprobs[i].first = i;
        idxNprobs[i].second = probs[i];
    }

    sort(idxNprobs.begin(), idxNprobs.end(), CompareIdxNVal); //decending order

    double norm=1;
    for( size_t i=0; i<idxNprobs.size(); i++ )
    {
        int c = ceil(((double)count)*idxNprobs[i].second/norm);

        count -= c;
        norm  -= idxNprobs[i].second;

        if( count <= 0 )
        {
            res[idxNprobs[i].first] = c+count;
            for( size_t j=i+1; j<idxNprobs.size(); j++ )
            {
                res[idxNprobs[j].first] = 0;
            }
            break;
        }
        else
        {
            res[idxNprobs[i].first] = c;
        }
    }
}

static void Normalize(vector<double> &prob)
{
    double norm = 0;
    for( size_t i=0; i<prob.size(); i++ ) norm += prob[i];
    if( norm==0 )
    {
        for( size_t i=0; i<prob.size(); i++ ) prob[i] = 1.0/(double)prob.size();
    }
    else
    {
        for( size_t i=0; i<prob.size(); i++ ) prob[i] /= norm;
    }    
}

static double reward(double dist)
{
    return 1/(1+exp(200*(dist-0.02)));
}

void VisModelBuilder::Sampling(     
    VisConvexHull vischulls[][NUM_OF_HYPS],
    vector<double> prob_objects,
    vector<pair<size_t,size_t> > &idxes_chosen,    
    pair<float,vector<VisConvexHull*> > &res )
{
    size_t n_objs = _models.size();
    if( idxes_chosen.size() == n_objs )
    {   
        vector<VisConvexHull*> chulls_test;
        for( size_t c=0; c<idxes_chosen.size(); c++ )
        {    
            size_t o = idxes_chosen[c].first;
            size_t h = idxes_chosen[c].second;
            res.second[o] = &vischulls[o][h];
            chulls_test.push_back(&vischulls[o][h]);            
        }
        VisModelSimulator simulator;
        simulator.SetupStaticScene(_plane_ground);

#if DEBUG_SIMULATOR
        res.first = simulator.Test(chulls_test, false);
#else
        res.first = simulator.Test(chulls_test);
#endif

        cout << res.first << ": ";
        for( size_t i=0; i<idxes_chosen.size(); i++ )
        {
            size_t o = idxes_chosen[i].first;
            cout << "[" << o << "]-" << idxes_chosen[i].second << " ";
        }
        cout << flush << endl;
        return;    
    } 
        
    // Setup with already chosen objects and hypotheses
    vector<PointCloud<PointXYZRGB>::Ptr> clouds_static;
    vector<VisConvexHull*> chulls_static;
    for( size_t i=0; i<idxes_chosen.size(); i++ )
    {
        size_t o = idxes_chosen[i].first;
        size_t h = idxes_chosen[i].second;
        prob_objects[o] = 0;
        chulls_static.push_back(&(vischulls[o][h]));
    }    
    Normalize(prob_objects);
    size_t o_chosen = PickRandom(prob_objects);
    for( size_t o=0; o<n_objs; o++ )
    {
        if( o==o_chosen ) continue;        
        bool isChosen = false;
        for( size_t i=0; i<idxes_chosen.size(); i++ )
        {
            size_t o2 = idxes_chosen[i].first;
            if( o==o2 )
            {
                isChosen = true; // already added as chull
                break;
            } 
        }
        if( isChosen==false ) clouds_static.push_back(_models[o].cloud);        
    }

    int count_rewards=1;
    float rewards[NUM_OF_HYPS];
    for( size_t h=0; h<NUM_OF_HYPS; h++ ) rewards[h] = 0;
    
    // Current Gravity
    vector<double> prob_hypotheses(NUM_OF_HYPS);
    for( size_t h=0; h<NUM_OF_HYPS; h++ )
    {
        VisModelSimulator simulator;
        simulator.SetupStaticScene(_plane_ground, clouds_static, chulls_static);
#if DEBUG_SIMULATOR
        float dist = simulator.Test(vischulls[o_chosen][h], false);        
#else
        float dist = simulator.Test(vischulls[o_chosen][h]);
#endif        
        rewards[h] += reward(dist);
    }
    
    for( size_t h=0; h<NUM_OF_HYPS; h++ )
    {
        prob_hypotheses[NUM_OF_HYPS] = rewards[h]/(float)count_rewards;
    } 

    Normalize(prob_hypotheses);
    size_t h_chosen = PickRandom(prob_hypotheses);
    idxes_chosen.push_back(pair<size_t, size_t>(o_chosen,h_chosen));

    Sampling( vischulls, prob_objects, idxes_chosen, res);
}

static bool CompareResult( 
    pair<float, vector<VisModelBuilder::VisConvexHull*> > p1,
    pair<float, vector<VisModelBuilder::VisConvexHull*> > p2  )
{
    return p1.first < p2.first;
}

static bool CompareProb( 
    pair<float, VisModelBuilder::VisConvexHull*> p1,
    pair<float, VisModelBuilder::VisConvexHull*> p2  )
{
    return p1.first > p2.first;
}

void VisModelBuilder::Sampling2(     
    vector<vector<pair<float,VisConvexHull*> > > &probs, 
    vector<VisConvexHull*> &best, 
    vector<VisConvexHull*> &most_stable  )
{
    int n_objs = _models.size();

    VisConvexHull vchull_tf[n_objs][5];
    int counts[n_objs][5];
    for( size_t o=0; o<n_objs; o++ )
    {
        for( size_t i=0; i<5; i++ )
        {
            probs[o][i].first = 0;
            counts[o][i] = 0;

            vchull_tf[o][i].cloud = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
            vchull_tf[o][i].cloud_hull = PointCloud<PointXYZRGB>::Ptr(new PointCloud<PointXYZRGB>);
            transform(*probs[o][i].second, vchull_tf[o][i], _models[o].tf);
        }        
    }

    int num_of_samples = pow(5,n_objs);
    vector<pair<float,vector<VisConvexHull*> > > res(num_of_samples);

    int hs[num_of_samples][n_objs];
#if DEBUG_SIMULATOR
#else
    #pragma omp parallel for
#endif
    for( int s=0; s<num_of_samples; s++ )
    {
        vector<VisConvexHull*> chulls_test;        
        for( size_t o=0; o<n_objs; o++ )
        {
            hs[s][o] = (int)(s / pow(5,o)) % 5;
            chulls_test.push_back(&vchull_tf[o][hs[s][o]]);         
        }

        VisModelSimulator simulator;
        simulator.SetupStaticScene(_plane_ground);
#if DEBUG_SIMULATOR
        res[s].first = simulator.Test(chulls_test, false);
#else
        res[s].first = simulator.Test(chulls_test);
#endif        
        res[s].second.resize(n_objs);
        for( size_t o=0; o<n_objs; o++ )
        {
            res[s].second[o] = probs[o][hs[s][o]].second;
        }
    }

    for( int s=0; s<num_of_samples; s++ )
    {
        float dist = res[s].first / (float)n_objs;
        double R = reward(dist);        
        for( size_t o=0; o<n_objs; o++ )
        {
            probs[o][hs[s][o]].first += R;
            counts[o][hs[s][o]]++;
        }
    }

    sort(res.begin(), res.end(), CompareResult);
    most_stable.resize(n_objs);
    for( size_t o=0; o<n_objs; o++ )
    {
        most_stable[o] = res[0].second[o];
    }

    for( size_t o=0; o<n_objs; o++ )
    {
        for( int i=0; i<5; i++ )
        {
            if( counts[o][i] > 0 ) probs[o][i].first /= (float)counts[o][i];
            else probs[o][i].first = 0;
        }
    }

    best.resize(n_objs);
    cout << "best: ";
    for( size_t o=0; o<n_objs; o++ )
    {
        double norm = 0;
        for( size_t h=0; h<NUM_OF_HYPS; h++ )
        {
            norm += probs[o][h].first;
        }
        for( size_t h=0; h<NUM_OF_HYPS; h++ )
        {
            probs[o][h].first /= norm;
        }
        sort(probs[o].begin(), probs[o].end(), CompareProb);

        cout << probs[o][0].second << " ";

        best[o] = probs[o][0].second;
    }
}

void VisModelBuilder::Sampling( 
    VisConvexHull vischulls[][NUM_OF_HYPS],
    vector<pair<float,vector<VisConvexHull*> > > &res,
    vector<vector<pair<float,VisConvexHull*> > > &probs, 
    vector<VisConvexHull*> &best, 
    vector<VisConvexHull*> &most_stable  )
{
    srand(0);
    int num_of_samples = 100;

    vector<VisModel>* models = &_models;    

    int n_objs = models->size();

    // Compute Prob(O=root)
    vector<double> probs_hypotheses[n_objs];
    vector<size_t> idxes_stable, idxes_unstable;
    for( size_t o=0; o<n_objs; o++ )
    {
        probs_hypotheses[o].resize(NUM_OF_HYPS);

        // Simulation
        bool stand_alone = false;        
        float dists[NUM_OF_HYPS];

#if DEBUG_SIMULATOR
#else
        #pragma omp parallel for
#endif
        for( size_t h=0; h<NUM_OF_HYPS; h++ )
        {
            VisModelSimulator simulator;
            vector<PointCloud<PointXYZRGB>::Ptr> clouds_static;
            for( size_t o2=0; o2<n_objs; o2++ )
            {
                if( o2==o ) continue;
                clouds_static.push_back((*models)[o2].cloud);
            }
            simulator.SetupStaticScene(_plane_ground, clouds_static);

#if DEBUG_SIMULATOR
            dists[h] = simulator.Test(vischulls[o][h], false);
#else
            dists[h] = simulator.Test(vischulls[o][h]);
#endif

            if( dists[h] <= 0.03 ) stand_alone = true;
            probs_hypotheses[o][h] = reward(dists[h]);

            cout << "o=" << o << ", h=" << h << ": " << dists[h] << ", " << probs_hypotheses[o][h] << flush << endl;
        }
        Normalize(probs_hypotheses[o]);

        if( stand_alone ) idxes_stable.push_back(o);
        else              idxes_unstable.push_back(o);
    }

    double eps;    
    if(        idxes_stable.size()==0 ) eps = 1.0;
    else if( idxes_unstable.size()==0 ) eps = 0.0;    
    else  eps = 1.0/(double)(1+2*idxes_stable.size()/idxes_unstable.size());
        
    vector<double> prob_objects(n_objs);
    for( size_t o=0; o<n_objs; o++ )
    {
        prob_objects[o] = 0;
    }
    for( size_t i=0; i<idxes_stable.size(); i++ )
    {        
        prob_objects[idxes_stable[i]] = (1-eps) / (double)idxes_stable.size();
    }    
    for( size_t i=0; i<idxes_unstable.size(); i++ )
    {        
        prob_objects[idxes_unstable[i]] = eps / (double)idxes_unstable.size();
    }
    Normalize(prob_objects);

    res.resize(num_of_samples);
#if DEBUG_SIMULATOR
#else
    #pragma omp parallel for
#endif
    for( size_t s=0; s<num_of_samples; s++ )
    {
        res[s].first = 0;
        res[s].second.resize(n_objs);

        size_t o = PickRandom(prob_objects);
        size_t h = PickRandom(probs_hypotheses[o]);

        vector<pair<size_t,size_t> > idxes_chosen;
        idxes_chosen.push_back(pair<uint32_t,uint32_t>(o,h));
        Sampling( vischulls, prob_objects, idxes_chosen, res[s]);
    }

    sort(res.begin(), res.end(), CompareResult);
    most_stable.resize(n_objs);
    for( size_t o=0; o<n_objs; o++ )
    {
        most_stable[o] = res[0].second[o];
    }

    vector<map<VisConvexHull*, size_t> > ptr2idx;
    probs.resize(n_objs);
    ptr2idx.resize(n_objs);
    int counts[n_objs][NUM_OF_HYPS];
    for( size_t o=0; o<n_objs; o++ )
    {
        ptr2idx[o].clear();
        probs[o].resize(NUM_OF_HYPS);
        for( size_t h=0; h<NUM_OF_HYPS; h++ )
        {
            counts[o][h] = 0;

            probs[o][h].first = 0;
            probs[o][h].second = &vischulls[o][h];

            ptr2idx[o].insert(pair<VisConvexHull*,size_t>(probs[o][h].second,h));
        }
    }

    for( size_t s=0; s<num_of_samples; s++ )
    {
        float dist = res[s].first / (float)n_objs;
        //float dist = res[s].first;
        double R = reward(dist);
        for( size_t o=0; o<n_objs; o++ )
        {
            size_t oh = ptr2idx[o].find(res[s].second[o])->second;
            
            //if( R > 0.5 )
            {
                probs[o][oh].first += R;
                counts[o][oh]++;
            }
        }        
    }

    for( size_t s=0; s<num_of_samples; s++ )
    {
        for( size_t o=0; o<n_objs; o++ )
        {
            size_t oh = ptr2idx[o].find(res[s].second[o])->second;
            if( counts[o][oh] > 0 ) probs[o][oh].first /= (float)counts[o][oh];
            else probs[o][oh].first = 0;
        }        
    }    

    best.resize(n_objs);
    cout << "best: ";
    for( size_t o=0; o<n_objs; o++ )
    {
        double norm = 0;
        for( size_t h=0; h<NUM_OF_HYPS; h++ )
        {
            norm += probs[o][h].first;
        }
        for( size_t h=0; h<NUM_OF_HYPS; h++ )
        {
            probs[o][h].first /= norm;
        }
        sort(probs[o].begin(), probs[o].end(), CompareProb);

        cout << probs[o][0].second << " ";

        best[o] = probs[o][0].second;
    }
    cout << endl;
#if 1
visualization::PCLVisualizer viewer; 
int vn[_models.size()*4];
int i=0;
for( size_t o=0; o<_models.size(); o++ )
{       
    float col = 1.0/(float)_models.size();
    viewer.createViewPort (col*o, 0.75, col*(o+1), 1.00, vn[i++]);
    viewer.createViewPort (col*o, 0.50, col*(o+1), 0.75, vn[i++]);
    viewer.createViewPort (col*o, 0.25, col*(o+1), 0.50, vn[i++]);
    viewer.createViewPort (col*o, 0.00, col*(o+1), 0.25, vn[i++]);
}
viewer.setWindowName("prob shape");
viewer.setSize(600,480);
viewer.setPosition(600,0);
viewer.setCameraPosition(-1,0,0,1,0,0,0,0,1);
viewer.setBackgroundColor (1, 1, 1);
i=0;
for( size_t o=0; o<_models.size(); o++ )
{
    cout << "o=" << o << ": ";
    int idxes[4] = {0, 1, 2, 3};
    for( size_t k=0; k<4; k++ )
    {
        int h = idxes[k];
        stringstream ss;
        ss << o << h;
        viewer.addPointCloud(_models[o].cloud, "cloud" + ss.str(), vn[i]);    
        viewer.addPolygonMesh(probs[o][h].second->polymesh, "polymesh" + ss.str(), vn[i]);
        /*
        viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_COLOR, 
                                                colors_vis[o][0]/255.,
                                                colors_vis[o][1]/255.,
                                                colors_vis[o][2]/255.,
                                                "polymesh3" + ss.str(), vn[i]);
        */
        cout << probs[o][h].first << " ";
        i++;
    }    
    cout << endl;
}
viewer.spin();
#endif


    VisModelSimulator simulator;
    simulator.SetupStaticScene(_plane_ground);
    cout << res[0].first << endl;
    cout << res[1].first << endl;
    float dist = simulator.Test(res[0].second);
    cout << res[0].first << ": ";
    for( size_t o=0; o<n_objs; o++ )
    {
        cout << res[0].second[o] << " ";
    }
    cout << endl;
    cout << dist << endl;
}

void VisModelBuilder::GenerateModels( VisConvexHull vischulls[][NUM_OF_HYPS] )
{
    srand(0);
    // Generate All Hypotheses    
    for( size_t o=0; o<_models.size(); o++ )
    {   
        size_t h=0;

        GenerateGravityModel2(o, vischulls[o][h]); h++;  
        GenerateTwoFacesModel(o, vischulls[o][h]); h++; 

        GenerateModel(o, vischulls[o][h], GEN_MINIMUM   | NO_NEWFACE); h++;

        GenerateModel(o, vischulls[o][h], GEN_MINIMUM   | GEN_GRAVFACE); h++;
        GenerateModel(o, vischulls[o][h], GEN_MAXIMUM_H | GEN_GRAVFACE); h++;
        GenerateModel(o, vischulls[o][h], GEN_MEDIUM    | GEN_GRAVFACE); h++;
        GenerateModel(o, vischulls[o][h], GEN_RANDOM    | GEN_GRAVFACE); h++;

        GenerateModel(o, vischulls[o][h], GEN_MAXIMUM_H | NO_NEWFACE); h++;
        GenerateModel(o, vischulls[o][h], GEN_MAXIMUM_V | NO_NEWFACE); h++;
        GenerateModel(o, vischulls[o][h], GEN_MEDIUM    | NO_NEWFACE); h++;
        GenerateModel(o, vischulls[o][h], GEN_RANDOM    | NO_NEWFACE); h++;

        GenerateModel(o, vischulls[o][h], GEN_MAXIMUM_H | GEN_NEWFACE); h++;
        GenerateModel(o, vischulls[o][h], GEN_MAXIMUM_V | GEN_NEWFACE); h++;
        GenerateModel(o, vischulls[o][h], GEN_MEDIUM    | GEN_NEWFACE); h++;
        GenerateModel(o, vischulls[o][h], GEN_RANDOM    | GEN_NEWFACE); h++;

        GenerateModel(o, vischulls[o][h], GEN_MAXIMUM_V | GEN_NEWFACE_MID); h++; 
        GenerateModel(o, vischulls[o][h], GEN_MAXIMUM_H | GEN_NEWFACE_MID); h++; 
        GenerateModel(o, vischulls[o][h], GEN_MEDIUM    | GEN_NEWFACE_MID); h++; 
        GenerateModel(o, vischulls[o][h], GEN_RANDOM    | GEN_NEWFACE_MID); h++; 
        
        GenerateModel(o, vischulls[o][h], GEN_MAXIMUM_V | GEN_NEWFACE_MAX); h++; 
        GenerateModel(o, vischulls[o][h], GEN_MAXIMUM_H | GEN_NEWFACE_MAX); h++; 
        GenerateModel(o, vischulls[o][h], GEN_MEDIUM    | GEN_NEWFACE_MAX); h++; 
        GenerateModel(o, vischulls[o][h], GEN_RANDOM    | GEN_NEWFACE_MAX); h++; 

        GenerateModel(o, vischulls[o][h], GEN_RANDOM    | GEN_NEWFACE    ); h++;     
        GenerateModel(o, vischulls[o][h], GEN_RANDOM    | GEN_NEWFACE_MID); h++; 
        GenerateModel(o, vischulls[o][h], GEN_RANDOM    | GEN_NEWFACE_MAX); h++; 

        GenerateModel(o, vischulls[o][h], GEN_RANDOM    | GEN_NEWFACE    ); h++;     
        GenerateModel(o, vischulls[o][h], GEN_RANDOM    | GEN_NEWFACE_MID); h++; 
        GenerateModel(o, vischulls[o][h], GEN_RANDOM    | GEN_NEWFACE_MAX); h++; 

        GenerateModel(o, vischulls[o][h], GEN_RANDOM    | GEN_NEWFACE    ); h++;     
        GenerateModel(o, vischulls[o][h], GEN_RANDOM    | GEN_NEWFACE_MID); h++; 
        GenerateModel(o, vischulls[o][h], GEN_RANDOM    | GEN_NEWFACE_MAX); h++; 

        //GenerateModel(o, vischulls[o][h], GEN_RANDOM    | GEN_NEWFACE    ); h++;             
        //GenerateModel(o, vischulls[o][h], GEN_RANDOM    | GEN_NEWFACE_MAX); h++;

        /*
        for( ; h<NUM_OF_HYPS; h++ )
        {            
            int flag = h%2==0 ? GEN_NEWFACE:NO_NEWFACE;            
            ;
        }
        */
    }
}

void VisModelBuilder::GenerateModel(size_t o, VisConvexHull &vischull, int flag)
{
    cout << "GenerateModel flag: " << hex << flag << dec << endl;

    VisModel &model = _models[o];

    PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);    
    PointCloud<PointXYZUncertain>::Ptr cloud_uncertain(new PointCloud<PointXYZUncertain>);
    copyPointCloud(*_cloud_uncertain, *cloud_uncertain);

    PointCloud<PointXYZRGB>::Ptr cloud_fig(new PointCloud<PointXYZRGB>);    
    *cloud_fig += *model.cloud;

    PointCloud<PointXYZRGB>::Ptr cloud_others(new PointCloud<PointXYZRGB>);
    for( size_t o_other=0; o_other<_models.size(); o_other++ )
    {
        if( o==o_other ) continue;

        *cloud_others += *_models[o_other].cloud;
    }    
    search::KdTree<PointXYZRGB> kdtree_others;
    kdtree_others.setInputCloud(cloud_others);

    // 1) Mirror observed faces
    vector<VisFace> faces_mirror;
    if( model.faces.size()<2 )
    {
        int flag_shape;
        if( (flag & MASK_GENSIZE) == GEN_MAXIMUM_V ||
            (flag & MASK_GENSIZE) == GEN_MAXIMUM_H )
        {
            flag_shape = MAXIMUM_SHAPE;
        }
        else if( (flag & MASK_GENSIZE) == GEN_MINIMUM )
        {
            flag_shape = MINIMUM_SHAPE;
        }
        else if( (flag & MASK_GENSIZE) == MEDIUM_SHAPE )
        {
            flag_shape = MEDIUM_SHAPE;            
        }
        else
        {
            flag_shape = RANDOM_SHAPE;
        }
        
        VisFace &face = model.faces[0];
        VisFace face_mirror;
        if( !GetRandomMirrorPoints( o, face.cloud, NULL, face.plane,
                                    face_mirror, cloud_uncertain, kdtree_others,
                                    flag_shape | FROM_ONEFACE, 0.02, cloud_fig) )
        {
           ROS_WARN_STREAM("??? Failed to get mirror points (from one face)");           
        }
        else
        {
            faces_mirror.push_back(face_mirror);

            *cloud += *face.cloud;
            *cloud += *face_mirror.cloud;            

            *cloud_fig += *face_mirror.cloud;
        }
    }
    else
    {
        int f_max = -1, f_min = -1;
        float vecz_max=-INFINITY, vecz_min=INFINITY;
        int flags_shape[model.faces.size()];
        for( size_t f=0; f<model.faces.size(); f++ )
        {
            VisFace &face = model.faces[f];            
            
            if( (flag & MASK_GENSIZE) == GEN_MAXIMUM_V )
            {
                flags_shape[f]= (abs(face.plane.coef[2])>COS_15) ? 
                                MAXIMUM_SHAPE : MINIMUM_SHAPE;                
            }
            else if( (flag & MASK_GENSIZE) == GEN_MAXIMUM_H )
            {
                flags_shape[f]= (abs(face.plane.coef[2])<COS_75) ? 
                                MAXIMUM_SHAPE : MINIMUM_SHAPE;                
            }
            else if( (flag & MASK_GENSIZE) == GEN_MINIMUM )
            {
                flags_shape[f]= MINIMUM_SHAPE;
            }            
            else
            {
                flags_shape[f]= RANDOM_SHAPE;
            }            

            if( vecz_max < abs(face.plane.coef[2]) )
            {
                f_max = f;
                vecz_max = abs(face.plane.coef[2]);
            }
            if( vecz_min > abs(face.plane.coef[2]) )
            {
                f_min = f;
                vecz_min = abs(face.plane.coef[2]);
            }
        }

        if( (flag & MASK_GENSIZE) == GEN_MAXIMUM_V )
        {
            flags_shape[f_max] = MAXIMUM_SHAPE;
        } 
        else if( (flag & MASK_GENSIZE) == GEN_MAXIMUM_H )
        {
            flags_shape[f_min] = MAXIMUM_SHAPE;
        }
        else if( (flag & MASK_GENSIZE) == MEDIUM_SHAPE )
        {
            flags_shape[f_min] = MEDIUM_SHAPE;            
        }

        for( size_t f=0; f<model.faces.size(); f++ )
        {
            VisFace &face = model.faces[f];
            VisFace face_mirror;
            if( !GetRandomMirrorPoints( o, face.cloud, NULL, face.plane,
                                        face_mirror, cloud_uncertain, kdtree_others,
                                        flags_shape[f] | FROM_ONEFACE, 0.01, cloud_fig) )
            {
               ROS_WARN_STREAM("??? Failed to get mirror points (from multi faces)");
            }
            else
            {
                faces_mirror.push_back(face_mirror);

                *cloud += *face.cloud;
                *cloud += *face_mirror.cloud;

                if( f_max == f ) *cloud_fig += *face_mirror.cloud;
            }
        }
    }      

    if( (flag & MASK_NEWFACE) != NO_NEWFACE )
    {
        if( (flag & MASK_NEWFACE) == GEN_GRAVFACE )
        {        
            VisFace face_mirror;                        
            if( !GetRandomMirrorPoints( o, cloud, NULL, _plane_ground,
                                        face_mirror, cloud_uncertain, kdtree_others,
                                        GRAVITY_SHAPE | FROM_ALL, 0, cloud_fig) )
            {
               ROS_WARN_STREAM("??? Failed to get mirror points (from multi faces)");
            }
            else
            {
                faces_mirror.push_back(face_mirror);
                *cloud += *face_mirror.cloud;
            }
        }
        else
        {
            PolygonMesh polymesh;
            ConvexHull<PointXYZRGB> chull;
            chull.setDimension(3);
            chull.setInputCloud (cloud);
            chull.reconstruct (polymesh);

            PointCloud<PointXYZRGB>::Ptr cloud_hull (new PointCloud<PointXYZRGB>);
            fromPCLPointCloud2(polymesh.cloud, *cloud_hull);

            // 2) Mirror generated faces
            // cluster the generated triangles based on their normal
            vector<Plane> planes_gen;
            vector<PointCloud<PointXYZRGB>::Ptr> clouds_plane_gen;
            vector<PointCloud<PointXYZRGB>::Ptr> clouds_plane_gen_mid;
            for( size_t v=0; v<polymesh.polygons.size(); v++ )
            {
                vector<uint32_t> &vertices = polymesh.polygons[v].vertices;

                PointXYZRGB &pt0 = cloud_hull->points[vertices[0]];
                PointXYZRGB &pt1 = cloud_hull->points[vertices[1]];
                PointXYZRGB &pt2 = cloud_hull->points[vertices[2]];

                Eigen::Vector3f vec1, vec2, normal;
                vec1 << pt0.x-pt1.x, pt0.y-pt1.y, pt0.z-pt1.z;        
                vec2 << pt0.x-pt2.x, pt0.y-pt2.y, pt0.z-pt2.z;
                normal = vec1.cross(vec2);
                normal = normal.normalized();

                Plane plane;
                GetPlane<PointXYZRGB>(normal, pt0, plane);

                bool exist=false;                
                for( size_t f=0; f<model.faces.size(); f++ )
                {            
                    if( IsSamePlane(plane,pt0, model.faces[f].plane, 
                                               model.faces[f].cloud_hull->points[0] ) || 
                        IsSamePlane(plane,pt0, faces_mirror[f].plane,
                                               faces_mirror[f].cloud_hull->points[0] ) )
                    {
                        exist = true;
                        break;
                    }            
                }                
                if( exist ) continue;

                for( size_t pl=0; pl<planes_gen.size(); pl++ )
                {
                    if( IsSamePlane(plane,pt0, planes_gen[pl], 
                                               clouds_plane_gen[pl]->points[0]) )
                    {
                        PointXYZRGB pt_center;
                        pt_center.x=0; pt_center.y=0; pt_center.z=0;
                        for( size_t p=0; p<vertices.size(); p++ )
                        {
                            PointXYZRGB &pt_add = cloud_hull->points[vertices[p]];
                            clouds_plane_gen[pl]->push_back(pt_add);
                            pt_center.x += pt_add.x;
                            pt_center.y += pt_add.y;
                            pt_center.z += pt_add.z; 
                        }
                        pt_center.x /= vertices.size();
                        pt_center.y /= vertices.size();
                        pt_center.z /= vertices.size();
                        clouds_plane_gen_mid[pl]->push_back(pt_center);

                        exist = true;
                        break;
                    }            
                }
                if( exist ) continue;

                planes_gen.push_back(plane);

                PointCloud<PointXYZRGB>::Ptr cloud_gen(new PointCloud<PointXYZRGB>);
                PointCloud<PointXYZRGB>::Ptr cloud_gen_mid(new PointCloud<PointXYZRGB>);
                PointXYZRGB pt_center;
                pt_center.x=0; pt_center.y=0; pt_center.z=0;
                for( size_t p=0; p<vertices.size(); p++ )
                {
                    PointXYZRGB &pt_add = cloud_hull->points[vertices[p]];
                    cloud_gen->push_back(pt_add);
                    pt_center.x += pt_add.x;
                    pt_center.y += pt_add.y;
                    pt_center.z += pt_add.z;
                }
                pt_center.x /= vertices.size();
                pt_center.y /= vertices.size();
                pt_center.z /= vertices.size();
                cloud_gen->push_back(pt_center);
                cloud_gen_mid->push_back(pt_center);

                clouds_plane_gen.push_back(cloud_gen);
                clouds_plane_gen_mid.push_back(cloud_gen_mid);
            }

        #if 1
            // generate new faces and mirror them
            int pl_max = -1;
            float area_max = -INFINITY;
            vector<VisFace> faces_gen_mirror;
            vector<VisFace> faces_gen(clouds_plane_gen.size());
            for( size_t pl=0; pl<faces_gen.size(); pl++ )
            {
                VisFace &face = faces_gen[pl];
                face.plane = planes_gen[pl];
                face.cloud = clouds_plane_gen[pl];
                face.cloud_hull = face.cloud;        
                        
                float area=0;        
                toPCLPointCloud2(*clouds_plane_gen[pl], face.polymesh.cloud);        
                for( size_t p=0; p<clouds_plane_gen[pl]->size(); p=p+3 )
                {
                    Vertices vertices;
                    vertices.vertices.resize(3);
                    vertices.vertices[0] = p;
                    vertices.vertices[1] = p+1;
                    vertices.vertices[2] = p+2;
                    face.polymesh.polygons.push_back(vertices);

                    Eigen::Vector3f va,vb;
                    va << (*clouds_plane_gen[pl])[p].x - (*clouds_plane_gen[pl])[p+1].x,
                          (*clouds_plane_gen[pl])[p].y - (*clouds_plane_gen[pl])[p+1].y,
                          (*clouds_plane_gen[pl])[p].z - (*clouds_plane_gen[pl])[p+1].z;
                    vb << (*clouds_plane_gen[pl])[p].x - (*clouds_plane_gen[pl])[p+2].x,
                          (*clouds_plane_gen[pl])[p].y - (*clouds_plane_gen[pl])[p+2].y,
                          (*clouds_plane_gen[pl])[p].z - (*clouds_plane_gen[pl])[p+2].z;
                    
                    area += va.cross(vb).norm();
                }            
                area /= 2;
                face.area = area;

                if( face.area > 0.001 )
                {
                    SACSegmentation<PointXYZRGB> seg;    
                    seg.setOptimizeCoefficients (true);
                    seg.setModelType (SACMODEL_PLANE);
                    seg.setMethodType (SAC_RANSAC);
                    seg.setDistanceThreshold (0.01);
                    seg.setInputCloud (face.cloud);
                    face.coefficients = ModelCoefficients::Ptr(new ModelCoefficients);
                    face.inliers = PointIndices::Ptr(new PointIndices);
                    seg.segment (*face.inliers, *face.coefficients);
                    face.plane.coef[0] = face.coefficients->values[0];
                    face.plane.coef[1] = face.coefficients->values[1];
                    face.plane.coef[2] = face.coefficients->values[2];
                    face.plane.coef[3] =-face.coefficients->values[3];
                    
                    bool orthogonal_to_oFace = true;
                    for( size_t f=0; f<model.faces.size(); f++ )
                    {
                        VisFace &face2 = model.faces[f];

                        if( abs(face.plane.coef[0]*face2.plane.coef[0] +
                                face.plane.coef[1]*face2.plane.coef[1] +
                                face.plane.coef[2]*face2.plane.coef[2]   ) > COS_75 )
                        {
                            orthogonal_to_oFace = false;
                            break;
                        }
                    }
                    if( orthogonal_to_oFace==false )continue;

                    if( area_max < area )
                    {
                        pl_max = pl;
                        area_max = area;
                    }

////////////////////////////////
                    VisFace &face = faces_gen[pl];            

                    Plane plane1, plane2;
                    plane1 = face.plane;
                    plane2.coef[0] = -plane1.coef[0];
                    plane2.coef[1] = -plane1.coef[1];
                    plane2.coef[2] = -plane1.coef[2];
                    plane2.coef[3] = -plane1.coef[3];

                    int flag_shape;
                    if(      (flag & MASK_NEWFACE) != GEN_NEWFACE     ) flag_shape = RANDOM_SHAPE;
                    else if( (flag & MASK_NEWFACE) != GEN_NEWFACE_MID ) flag_shape = MEDIUM_SHAPE;
                    else if( (flag & MASK_NEWFACE) != GEN_NEWFACE_MAX ) flag_shape = MAXIMUM_SHAPE;                

                    // mirror maximum faces            
                    VisFace face_mirror1, face_mirror2;
                    bool succ = false;
                    if( !GetRandomMirrorPoints( o, 
                                                clouds_plane_gen[pl], clouds_plane_gen_mid[pl], 
                                                plane1, face_mirror1, cloud_uncertain, kdtree_others,
                                                flag_shape | FROM_ALL, 0, cloud_fig        ))
                    {
                       ROS_WARN_STREAM("??? Failed to get mirror points (gen face 1)");                   
                    }
                    else
                    {
                        faces_gen_mirror.push_back(face_mirror1);
                        *cloud += *face_mirror1.cloud_hull;
                    }
                    if( !GetRandomMirrorPoints( o, 
                                                clouds_plane_gen[pl], clouds_plane_gen_mid[pl], 
                                                plane2, face_mirror2, cloud_uncertain, kdtree_others,
                                                flag_shape | FROM_ALL, 0, cloud_fig        ))
                    {
                       ROS_WARN_STREAM("??? Failed to get mirror points (gen face 2)");                   
                    }
                    else
                    {
                        faces_gen_mirror.push_back(face_mirror2);
                        *cloud += *face_mirror2.cloud_hull;
                    }   
////////////////////////////////
                }            
            }
#if 0
            if( pl_max > -1 )
            {
                size_t pl = pl_max;
                VisFace &face = faces_gen[pl];            

                Plane plane1, plane2;
                plane1 = face.plane;
                plane2.coef[0] = -plane1.coef[0];
                plane2.coef[1] = -plane1.coef[1];
                plane2.coef[2] = -plane1.coef[2];
                plane2.coef[3] = -plane1.coef[3];

                int flag_shape;
                if(      (flag & MASK_NEWFACE) != GEN_NEWFACE     ) flag_shape = RANDOM_SHAPE;
                else if( (flag & MASK_NEWFACE) != GEN_NEWFACE_MID ) flag_shape = MEDIUM_SHAPE;
                else if( (flag & MASK_NEWFACE) != GEN_NEWFACE_MAX ) flag_shape = MAXIMUM_SHAPE;                

                // mirror maximum faces            
                VisFace face_mirror1, face_mirror2;
                bool succ = false;
                if( !GetRandomMirrorPoints( o, 
                                            clouds_plane_gen[pl], clouds_plane_gen_mid[pl], 
                                            plane1, face_mirror1, cloud_uncertain, kdtree_others,
                                            flag_shape | FROM_ALL, 0, cloud_fig        ))
                {
                   ROS_WARN_STREAM("??? Failed to get mirror points (gen face 1)");                   
                }
                else
                {
                    faces_gen_mirror.push_back(face_mirror1);
                    *cloud += *face_mirror1.cloud_hull;
                }
                if( !GetRandomMirrorPoints( o, 
                                            clouds_plane_gen[pl], clouds_plane_gen_mid[pl], 
                                            plane2, face_mirror2, cloud_uncertain, kdtree_others,
                                            flag_shape | FROM_ALL, 0, cloud_fig        ))
                {
                   ROS_WARN_STREAM("??? Failed to get mirror points (gen face 2)");                   
                }
                else
                {
                    faces_gen_mirror.push_back(face_mirror2);
                    *cloud += *face_mirror2.cloud_hull;
                }            
            }
#endif
        }
#endif 
    }

    //*cloud += *model.cloud;
    GenerateVischull(*cloud, vischull);
    vischull.tf_cur2init = model.tf.inverse();

#if DEBUG_MIRROR
if( o==2)
{
visualization::PCLVisualizer viewer; 
int v1, v2, v3;
viewer.createViewPort (0.0, 0.0, 0.3, 1.0, v1);
viewer.createViewPort (0.3, 0.0, 0.6, 1.0, v2);
viewer.createViewPort (0.6, 0.0, 1.0, 1.0, v3);
viewer.setWindowName("debug");
viewer.setSize(600,480);
viewer.setPosition(600,0);
viewer.setCameraPosition(-1,0,0,1,0,0,0,0,1);
viewer.setBackgroundColor (1, 1, 1);
viewer.addPointCloud(cloud, "cloud1", v1);
viewer.addPointCloud(cloud, "cloud2", v2);
viewer.addPointCloud(cloud, "cloud3", v3);
viewer.addPolygonMesh(vischull.polymesh, "polymesh3", v3);
//viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_COLOR, 
//                                        1,0,0, "polymesh3", v3);
viewer.spin();
}
#endif
}

void VisModelBuilder::GenerateTwoFacesModel(size_t o, VisConvexHull &vischull)
{
    GenerateModel(o, vischull, GEN_MINIMUM | GEN_GRAVFACE);


#if 0
    //GetConvexHull(o, VisModelBuilder::C_GRAVITYONEFACE_TWOFACES, vischull);
    GetConvexHull(o, C_GRAVITY | C_ONEFACE, vischull);
    vischull.tf_cur2init = _models[o].tf.inverse();    
#endif
}

void VisModelBuilder::GetConvexHull( size_t o, ENUM_Criteria criteria, 
                                     VisConvexHull &vischull )
{
    VisModel &model = _models[o];

    PointCloud<PointXYZRGB>::Ptr cloud (new PointCloud<PointXYZRGB>);
    *cloud += *model.cloud;

    // Add Uncertain cloud
    PointCloud<PointXYZRGB>::Ptr cloud_empty (new PointCloud<PointXYZRGB>);
    PointCloud<PointXYZRGB>::Ptr cloud_other (new PointCloud<PointXYZRGB>);
    for( size_t p=0; p<_cloud_uncertain->size(); p++ )
    {
        PointXYZUncertain &pt_uncertain = _cloud_uncertain->points[p];
        if( pt_uncertain.type[o] != EMPTY )
        {
            PointXYZRGB pt;
            pt.x = pt_uncertain.x; pt.y = pt_uncertain.y; pt.z = pt_uncertain.z; 
            pt.r = 0; pt.g = 0; pt.b = 0;
            
            ENUM_Criteria flag=C_NONE;
            if( pt_uncertain.gravity[o] )
            {              
                flag |= C_GRAVITY;
                pt.b = 255;
            }

            if( pt_uncertain.n_face_prjs[o] >= 2 )
            {
                flag |= C_TWOFACES;
                pt.g = 255;
            }
            else if ( pt_uncertain.n_face_prjs[o] == 1 )
            {
                flag |= C_ONEFACE;
                pt.g = 255/2;
            }

            /*
            if( ((flag&C_GRAVITY) && (flag&C_ONEFACE)) || 
                (flag&C_TWOFACES)==C_TWOFACES )
            {
                flag = C_GRAVITYONEFACE_TWOFACES;
            }
            */

            if( (flag & criteria) == criteria )
            {
                pt.r = 100; pt.g=100; pt.b=100;
                cloud->push_back(pt);
            }
            else
            {
                cloud_other->push_back(pt);
            }
        }
#if 0
        else if ( pt_uncertain.type == EMPTY )
        {
            PointXYZRGB pt;
            pt.x = pt_uncertain.x; pt.y = pt_uncertain.y; pt.z = pt_uncertain.z; 
            pt.r = 255; pt.g = 255; pt.b = 255;
            cloud_empty->push_back(pt);   
        }
#endif
    }

    GenerateVischull(*cloud, vischull);
    
#if 0
int v1, v2;
visualization::PCLVisualizer viewer; 
viewer.createViewPort (0.00, 0.0, 0.50, 1.0, v1);
viewer.createViewPort (0.50, 0.0, 1.00, 1.0, v2);
viewer.setWindowName("prob");
viewer.setSize(600,480);
viewer.setPosition(600,0);
viewer.setCameraPosition(-1,0,0,1,0,0,0,0,1);
viewer.setBackgroundColor (1, 1, 1);
viewer.addPointCloud(cloud, "object", v1);
viewer.addPointCloud(cloud_empty, "empty", v2);
viewer.addPointCloud(cloud_other, "other", v2);
viewer.spin();
#endif
}

vector<Plane>& VisModelBuilder::GetRemovedPlanes()
{
    return _planes_bg;
}

static
void BuildObjectsGraph( rl_msgs::SegmentationScene &segscene,
                        multimap<uint32_t,uint32_t> &edges   )
{
    PointCloud<PointXYZL>::Ptr cloud_all(new PointCloud<PointXYZL>);
    for( size_t o=0; o<segscene.objects.size(); o++ )
    {
        PointCloud<PointXYZRGB>::Ptr cloud_local(new PointCloud<PointXYZRGB>);
        fromRLMsg(segscene.objects[o], cloud_local);

        for( size_t p=0; p<cloud_local->size(); p++ )
        {
            PointXYZL pt;
            pt.x = cloud_local->points[p].x;
            pt.y = cloud_local->points[p].y;
            pt.z = 0;
            pt.label = o;

            cloud_all->push_back(pt);
        }
    }
    
    KdTreeFLANN<PointXYZL> kdtree;
    kdtree.setInputCloud (cloud_all);
    set<pair<uint32_t,uint32_t> > set_edges;
    for( size_t p=0; p<cloud_all->size(); p++ )
    {
        vector<int> idxes;
        vector<float> dists;
        PointXYZL &pt = cloud_all->points[p];
        uint32_t label1 = pt.label;
        if( kdtree.radiusSearch(pt, 0.01, idxes, dists) > 0 )
        {
            for( size_t i=0; i<idxes.size(); i++ )
            {                
                uint32_t label2 = cloud_all->points[idxes[i]].label;
                if( label1 != label2 )
                {
                    set_edges.insert(pair<uint32_t,uint32_t>(label1, label2));
                    set_edges.insert(pair<uint32_t,uint32_t>(label2, label1));
                }
            }
        }
    }
    
    for( set<pair<uint32_t,uint32_t> >::iterator it_set=set_edges.begin();
         it_set != set_edges.end(); it_set++ )
    {
        edges.insert(pair<uint32_t,uint32_t>(it_set->first,it_set->second));
        edges.insert(pair<uint32_t,uint32_t>(it_set->second,it_set->first));
    }
}

void VisModelBuilder::Update( Mat &image, Mat &depth, 
                              PointCloud<PointXYZRGB>::Ptr cloud,
                              std::vector<float> camera_RT )
{   
    if( camera_RT.size()==0 ) camera_RT = _camera_RT;
    else                      SetCameraRT(camera_RT);    
    
    mtx_update_2d.lock();
    mtx_update_3d.lock();
    if( _segscenes.size() > 0 )
    {
        rl_msgs::SegmentationScene segscene;
        thread t_seg3d( ThreadSegment, 
                        _cloudSeg,&image,cloud,&_param,&_camera_K,&camera_RT,
                        &segscene                  );

        // Update image and depth
        Mat image_seg;
        thread t_seg2d(ThreadQuickShift, &image, &image_seg);
        thread t_sift(ThreadSiftTrack, _imgTracker, &image);
        _depths.push_back(depth);
        t_seg2d.join();
        _images_seg.push_back(image_seg);
        t_sift.join();

        vector<vector<Point2f> > points_track;
        _imgTracker->Track(points_track);

        // Copy to local
        vector<Mat> depths(_depths.size());
        copy(_depths.begin(), _depths.end(), depths.begin());
        _depths.clear();
        vector<Mat> images_seg(_images_seg.size());
        copy(_images_seg.begin(), _images_seg.end(), images_seg.begin());
        _images_seg.clear();

        mtx_update_2d.unlock(); // can update image & depth

        t_seg3d.join();
        _segscenes.push_back(segscene);

        // Track segments using 2d segmentations
        multimap<size_t,size_t> beg2end, end2beg;
        vector<CloudTracker::Matrix4> TFs;
        vector<set<pair<size_t,size_t> > > traj;
        //rl_msgs::SegmentationScene &segscene_prev = _segscenes.front();
        rl_msgs::SegmentationScene &segscene_prev = _segscenes[_segscenes.size()-2];

        _cloudTracker->Track( segscene_prev, segscene,
                              depths, images_seg, points_track, traj,
                              beg2end, end2beg, TFs                  );
        
        // Don't trust tracker... T_T
        multimap<size_t,size_t> end2beg2;
        for( size_t e=0; e<segscene.objects.size(); e++ )
        {
            size_t b_min = 0;
            float dist_min = INFINITY;
            for( size_t b=0; b<segscene_prev.objects.size(); b++ )            
            {
                float dist = 
                (segscene_prev.objects[b].center.x-segscene.objects[e].center.x)*
                (segscene_prev.objects[b].center.x-segscene.objects[e].center.x)+
                (segscene_prev.objects[b].center.y-segscene.objects[e].center.y)*
                (segscene_prev.objects[b].center.y-segscene.objects[e].center.y)+
                (segscene_prev.objects[b].center.z-segscene.objects[e].center.z)*
                (segscene_prev.objects[b].center.z-segscene.objects[e].center.z);
                if( dist_min > dist )
                {
                    b_min = b;
                    dist_min = dist;
                } 
            }
            end2beg2.insert(pair<size_t,size_t>(e,b_min));

            TFs[e] << 1, 0, 0, segscene.objects[e].center.x-segscene_prev.objects[b_min].center.x,
                      0, 1, 0, segscene.objects[e].center.y-segscene_prev.objects[b_min].center.y,
                      0, 0, 1, segscene.objects[e].center.z-segscene_prev.objects[b_min].center.z,
                      0, 0, 0, 1;
        }

        AddBGPlane(segscene);
        //UpdateVisModel(segscene, end2beg, TFs);
        UpdateVisModel(segscene, end2beg2, TFs);
        //_segscenes.pop();        
    }
    else
    {
        rl_msgs::SegmentationScene segscene;
        thread t_seg3d( ThreadSegment, 
                      _cloudSeg, &image, cloud, &_param, &_camera_K, &camera_RT,
                      &segscene                  );

        Mat image_seg;
        thread t_seg2d(ThreadQuickShift, &image, &image_seg);
        thread t_sift(ThreadSiftTrack, _imgTracker, &image);
        _depths.push_back(depth);    
        t_seg2d.join();
        _images_seg.push_back(image_seg);
        t_sift.join();

        mtx_update_2d.unlock(); // can update image & depth

        t_seg3d.join();
        _segscenes.push_back(segscene);

        AddBGPlane(segscene);
        AddVisModel(segscene);
    }
    mtx_update_3d.unlock();

    _obj_edges.clear();
    BuildObjectsGraph(_segscenes[_segscenes.size()-1], _obj_edges);

#if 0
visualization::PCLVisualizer viewer; 
int v1, v2, v3;
viewer.createViewPort (0.00, 0.00, 0.33, 1.00, v1);
viewer.createViewPort (0.33, 0.00, 0.66, 1.00, v2);
viewer.createViewPort (0.66, 0.00, 0.99, 1.00, v3);
viewer.setWindowName("debug");
viewer.setSize(600,480);
viewer.setPosition(600,0);
viewer.setCameraPosition(-1,0,0,1,0,0,0,0,1);
viewer.setBackgroundColor (1, 1, 1);
viewer.addPointCloud(cloud, "cloud",v1);
DrawSegmentation(0,viewer,v2,v3);
viewer.spin();
#endif
}

void VisModelBuilder::VisModelSimulator::SetupStaticScene(
    Plane &plane_ground,
    const vector<PointCloud<PointXYZRGB>::Ptr> &clouds, 
    const vector<VisConvexHull*> &chulls     )
{
    ExitWorld();
    InitWorld();

    AddPlaneShape( plane_ground );    
    SetGravity(btVector3( -plane_ground.coef[0],
                          -plane_ground.coef[1],
                          -plane_ground.coef[2]  ));

    for( size_t i=0; i<clouds.size(); i++ )
    {
        PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);        
        copyPointCloud(*clouds[i], *cloud);
        
        PointXYZRGB pt_min, pt_max;
        getMinMax3D(*cloud, pt_min, pt_max);
        double cx = (pt_max.x+pt_min.x)/2, 
               cy = (pt_max.y+pt_min.y)/2, 
               cz = (pt_max.z+pt_min.z)/2;
        for( size_t p=0; p<cloud->size(); p++ )
        {
            cloud->points[p].x -= cx;
            cloud->points[p].y -= cy;
            cloud->points[p].z -= cz;
        }
        
        float mass=0; // add model as static object (mass=0)
        AddConvexHullShape(*cloud, btVector3(cx,cy,cz), mass); 
    }

    for( size_t i=0; i<chulls.size(); i++ )
    {
        float mass=0; // add model as static object (mass=0)
        AddConvexHullShape( *chulls[i]->cloud_hull, 
                            btVector3(chulls[i]->cx,
                                      chulls[i]->cy,
                                      chulls[i]->cz), mass );
    }
}

float VisModelBuilder::VisModelSimulator::Test(
    VisConvexHull &vischull, bool show)
{
    float dist = 0;
    AddConvexHullShape( *vischull.cloud_hull, 
                        btVector3(vischull.cx,vischull.cy,vischull.cz), 1);

#if DEBUG_SIMULATOR
    if( show )
    {
        SpinInit();
        ResetCamera( 0.25,-90,45, 0.12,0,0 );
        dist = SpinUntilStable();
        SpinExit();
    }
    else
    {
        dist = SpinUntilStable();
    }
#else
    dist = SpinUntilStable();
#endif

    RemoveLastRigidBody();
    return dist;
}

float VisModelBuilder::VisModelSimulator::Test(
    vector<VisConvexHull*> &vischulls, bool show)
{
    float dist = 0;
    for(size_t i=0; i<vischulls.size(); i++)
    {
        VisConvexHull &vischull = *vischulls[i];
        AddConvexHullShape( *vischull.cloud_hull, 
                            btVector3(vischull.cx,vischull.cy,vischull.cz), 1);
    }
    
#if DEBUG_SIMULATOR
    if( show )
    {
        SpinInit();
        ResetCamera( 0.25,-90,45, 0.12,0,0 );
        dist = SpinUntilStable();
        SpinExit();
    }
    else
    {
        dist = SpinUntilStable();
    }
#else
    dist = SpinUntilStable();
#endif

    for(size_t i=0; i<vischulls.size(); i++)
    {
        RemoveLastRigidBody();
    }
    return dist;
}

void VisModelBuilder::GetSegmentation( int idx, PointCloud<PointXYZRGB>::Ptr cloud)
{
    rl_msgs::SegmentationScene &segscene = _segscenes[_segscenes.size()-1];
    fromRLMsg(segscene.objects[idx], cloud);
}

void VisModelBuilder::DrawSegmentation( int idx, 
                                        visualization::PCLVisualizer &viewer, 
                                        int v1, int v2)
{
    rl_msgs::SegmentationScene &segscene = _segscenes[idx];
    if( v1!=-1 ) AddSegmentationObjects(segscene.objects,  viewer, v1, _tf_cam2world);
    if( v2!=-1 ) AddSegmentationObjects2(segscene.objects, viewer, v2, _tf_cam2world);
}