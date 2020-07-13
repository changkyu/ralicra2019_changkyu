#include <iostream>
#include <vector>
#include <stdlib.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>

#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <geometry_msgs/Point.h>
#include "rl_msgs/seg_scene_srv.h"
#include "rl_msgs/rl_msgs_conversions.hpp"
#include "rl_msgs/rl_msgs_visualization.hpp"

#include "segmentation/seg_lccp_2Dseg.hpp"
#include "random_push/pusher.hpp"
#include "camera_recorder/camera_recorder.hpp"

using namespace std;
using namespace pcl;
#define COS_30 (0.86602540378443860)

static vector<float> camera_RT;
static Eigen::Matrix4f tf_cam2world;
static ros::ServiceClient clt_lccp_2Dseg;
static float hand_length=0;
static float effector_length=0;
static float effector_width=0;
static float action_distance=0;
static struct
{
    float min_x;
    float max_x;
    float min_y;
    float max_y;
    float min_z;
    float max_z;
} workspace;
static string outdir;

static
void PrintParam()
{
    ROS_INFO_STREAM( endl
    << "effector_width: " << effector_width << endl
    << "effector_length: " << effector_length - hand_length << endl
    << "hand_length: " << hand_length << endl
    << "action_distance: " << action_distance << endl
    << "workspace: x[" << workspace.min_x << "," << workspace.max_x << "]" << endl
    <<           " y[" << workspace.min_y << "," << workspace.max_y << "]" << endl
    <<           " z[" << workspace.min_z << "," << workspace.max_z << "]" << endl
    << "outdir: " << outdir << endl
    << "camera_RT:" << endl
    <<camera_RT[0] <<","<<camera_RT[1] <<","<<camera_RT[2] <<","<<camera_RT[3] <<endl
    <<camera_RT[4] <<","<<camera_RT[5] <<","<<camera_RT[6] <<","<<camera_RT[7] <<endl
    <<camera_RT[8] <<","<<camera_RT[9] <<","<<camera_RT[10]<<","<<camera_RT[11]<<endl
    <<camera_RT[12]<<","<<camera_RT[13]<<","<<camera_RT[14]<<","<<camera_RT[15]<<endl
    << std::endl
    );
}

static
void ParseParam(ros::NodeHandle nh)
{
    nh.param<float>("/random_push/hand_length", hand_length, 0.13);
    nh.param<float>("/random_push/effector_length", effector_length, 0.43);
    effector_length += hand_length;

    nh.param<float>("/random_push/effector_width",  effector_width,  0.01);
    nh.param<float>("/random_push/action_distance", action_distance, 0.05);
    nh.param<float>("/random_push/min_x", workspace.min_x, -1.00 );
    nh.param<float>("/random_push/max_x", workspace.max_x,  1.00 );
    nh.param<float>("/random_push/min_y", workspace.min_y, -1.00 );
    nh.param<float>("/random_push/max_y", workspace.max_y,  1.00 );
    nh.param<float>("/random_push/min_z", workspace.min_z, -0.25 );
    nh.param<float>("/random_push/max_z", workspace.max_z,  0.70 );
    workspace.min_z += effector_length;
    
    nh.param<string>("/random_push/outdir",outdir,string(getenv("HOME"))+"/save/");
    nh.getParam("/random_push/cam_info/extrinsic",camera_RT);

    tf_cam2world << camera_RT[0],  camera_RT[1],  camera_RT[2],  camera_RT[3], 
                    camera_RT[4],  camera_RT[5],  camera_RT[6],  camera_RT[7], 
                    camera_RT[8],  camera_RT[9],  camera_RT[10], camera_RT[11], 
                    camera_RT[12], camera_RT[13], camera_RT[14], camera_RT[15];
}

typedef struct PushAction
{
    geometry_msgs::Point pt_start;
    geometry_msgs::Point pt_end;
} PushAction;

static
void Transform(geometry_msgs::Vector3 &normal_in, geometry_msgs::Vector3 &normal_out, vector<float> &RT)
{
    normal_out.x = normal_in.x*RT[0] + normal_in.y*RT[1] + normal_in.z*RT[2];
    normal_out.y = normal_in.x*RT[4] + normal_in.y*RT[5] + normal_in.z*RT[6];
    normal_out.z = normal_in.x*RT[8] + normal_in.y*RT[9] + normal_in.z*RT[10];
}

static
void Transform(geometry_msgs::Point &pt_in, geometry_msgs::Point &pt_out, vector<float> &RT)
{
    pt_out.x = pt_in.x*RT[0] + pt_in.y*RT[1] + pt_in.z*RT[2] + RT[3];
    pt_out.y = pt_in.x*RT[4] + pt_in.y*RT[5] + pt_in.z*RT[6] + RT[7];
    pt_out.z = pt_in.x*RT[8] + pt_in.y*RT[9] + pt_in.z*RT[10] + RT[11];
}

static
bool IsValidAction( PushAction &action, KdTreeFLANN<PointXYZRGB> &kdtree )
{
    PointXYZ pt_init(0.5,0.0,0.7 - effector_length);
    pt_init.x = action.pt_start.x;
    pt_init.y = action.pt_start.y;
    pt_init.z = action.pt_start.z + effector_length + 0.1;

    bool valid_ws = (action.pt_start.x > workspace.min_x) && 
                    (action.pt_end.x   > workspace.min_x) &&
                    (action.pt_start.x < workspace.max_x) && 
                    (action.pt_end.x   < workspace.max_x) &&
                    (action.pt_start.y > workspace.min_y) && 
                    (action.pt_end.y   > workspace.min_y) &&
                    (action.pt_start.y < workspace.max_y) && 
                    (action.pt_end.y   < workspace.max_y) &&
                    (action.pt_start.z > workspace.min_z) && 
                    (action.pt_end.z   > workspace.min_z) &&
                    (action.pt_start.z < workspace.max_z) && 
                    (action.pt_end.z   < workspace.max_z);

    if( valid_ws==false ) return false;

    vector<int> idxes(1);
    vector<float> dists(1);

    vector<PointXYZRGB> way_pts;
    PointXYZ vec;
    vec.x = action.pt_start.x - pt_init.x;
    vec.y = action.pt_start.y - pt_init.y;
    vec.z = action.pt_start.z - pt_init.z;
    double norm = sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
    vec.x /= norm; vec.y /= norm; vec.z /= norm;
    
    PointXYZRGB pt;
    pt.x = pt_init.x + vec.x * 0.01; 
    pt.y = pt_init.y + vec.y * 0.01; 
    pt.z = pt_init.z + vec.z * 0.01; 

    PointXYZ vec_pt2end;
    vec_pt2end.x = action.pt_start.x - pt.x;
    vec_pt2end.y = action.pt_start.y - pt.y;
    vec_pt2end.z = action.pt_start.z - pt.z;

    while( (vec.x*vec_pt2end.x + vec.y*vec_pt2end.y + vec.z*vec_pt2end.z) > effector_width*2 )
    {
        kdtree.nearestKSearch(pt, 1, idxes, dists);
        if( dists[0] < effector_width*effector_width )
        {
            return false;
        }
        
        pt.x = pt.x + vec.x * 0.01; 
        pt.y = pt.y + vec.y * 0.01; 
        pt.z = pt.z + vec.z * 0.01; 

        vec_pt2end.x = action.pt_start.x - pt.x;
        vec_pt2end.y = action.pt_start.y - pt.y;
        vec_pt2end.z = action.pt_start.z - pt.z;
    }

    return true;
}

static
void GeneratePushActions( rl_msgs::SegmentationScene &segscene, 
                          vector<PushAction> &actions_out      )
{
    vector<PushAction> actions;
    PointCloud<PointXYZRGB>::Ptr cloud_all(new PointCloud<PointXYZRGB>);
    for( size_t o=0; o<segscene.objects.size(); o++ )
    {
        rl_msgs::SegmentationObject &object = segscene.objects[o];

        PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
        fromROSMsg(object.cloud, *cloud);
        PointCloud<PointXYZRGB>::Ptr cloud_tf(new PointCloud<PointXYZRGB>);
        transformPointCloud(*cloud, *cloud_tf, tf_cam2world);
        *cloud_all += *cloud_tf;

        for( size_t f=0; f<object.faces.size(); f++ )
        {
            rl_msgs::SegmentationFace &face = object.faces[f];
            geometry_msgs::Point center;
            geometry_msgs::Vector3 normal;
            geometry_msgs::Vector3 axis1;
            geometry_msgs::Vector3 axis2;
            double axis1_size = face.sizes[0];
            double axis2_size = face.sizes[1];
            Transform(face.center, center, camera_RT);
            Transform(face.normal, normal, camera_RT);
            Transform(face.vecs_axis[0], axis1, camera_RT);
            Transform(face.vecs_axis[1], axis2, camera_RT);

            double norm = sqrt(normal.x*normal.x + normal.y*normal.y);
            if( normal.z < COS_30 )
            {

                /*
                PushAction pa;
                pa.pt_start.x = center.x + normal.x*0.01;
                pa.pt_start.y = center.y + normal.y*0.01;
                pa.pt_start.z = center.z + normal.z*0.01;
                pa.pt_end.x = pa.pt_start.x - normal.x/norm * 0.05;
                pa.pt_end.y = pa.pt_start.y - normal.y/norm * 0.05;
                pa.pt_end.z = pa.pt_start.z;            
                actions.push_back(pa);
                */

                PushAction pa2;
                pa2.pt_start.x = center.x + normal.x*0.03 + axis1.x * axis1_size/2;
                pa2.pt_start.y = center.y + normal.y*0.03 + axis1.y * axis1_size/2;
                pa2.pt_start.z = center.z + normal.z*0.03 + axis1.z * axis1_size/2;
                pa2.pt_end.x = pa2.pt_start.x - normal.x/norm * action_distance;
                pa2.pt_end.y = pa2.pt_start.y - normal.y/norm * action_distance;
                pa2.pt_end.z = pa2.pt_start.z;
                actions.push_back(pa2);

                PushAction pa3;
                pa3.pt_start.x = center.x + normal.x*0.03 - axis1.x * axis1_size/2;
                pa3.pt_start.y = center.y + normal.y*0.03 - axis1.y * axis1_size/2;
                pa3.pt_start.z = center.z + normal.z*0.03 - axis1.z * axis1_size/2;
                pa3.pt_end.x = pa3.pt_start.x - normal.x/norm * action_distance;
                pa3.pt_end.y = pa3.pt_start.y - normal.y/norm * action_distance;
                pa3.pt_end.z = pa3.pt_start.z;            
                actions.push_back(pa3);

                PushAction pa4;
                pa4.pt_start.x = center.x + normal.x*0.03 + axis2.x * axis2_size/2;
                pa4.pt_start.y = center.y + normal.y*0.03 + axis2.y * axis2_size/2;
                pa4.pt_start.z = center.z + normal.z*0.03 + axis2.z * axis2_size/2;
                pa4.pt_end.x = pa4.pt_start.x - normal.x/norm * action_distance;
                pa4.pt_end.y = pa4.pt_start.y - normal.y/norm * action_distance;
                pa4.pt_end.z = pa4.pt_start.z;
                actions.push_back(pa4);

                PushAction pa5;
                pa5.pt_start.x = center.x + normal.x*0.03 - axis2.x * axis2_size/2;
                pa5.pt_start.y = center.y + normal.y*0.03 - axis2.y * axis2_size/2;
                pa5.pt_start.z = center.z + normal.z*0.03 - axis2.z * axis2_size/2;
                pa5.pt_end.x = pa5.pt_start.x - normal.x/norm * action_distance;
                pa5.pt_end.y = pa5.pt_start.y - normal.y/norm * action_distance;
                pa5.pt_end.z = pa5.pt_start.z;            
                actions.push_back(pa5);

            }

            double norm1 = sqrt(axis1.x*axis1.x + axis1.y*axis1.y);
            double norm2 = sqrt(axis2.x*axis2.x + axis2.y*axis2.y);

            PushAction pa6;
            pa6.pt_start.x = center.x + axis1.x * (axis1_size+0.03) - normal.x*0.02;
            pa6.pt_start.y = center.y + axis1.y * (axis1_size+0.03) - normal.y*0.02;
            pa6.pt_start.z = center.z + axis1.z * (axis1_size+0.03) - normal.z*0.02;
            pa6.pt_end.x = pa6.pt_start.x - axis1.x/norm1 * action_distance;
            pa6.pt_end.y = pa6.pt_start.y - axis1.y/norm1 * action_distance;
            pa6.pt_end.z = pa6.pt_start.z;            
            actions.push_back(pa6);

            PushAction pa7;
            pa7.pt_start.x = center.x - axis1.x * (axis1_size+0.03) - normal.x*0.02;
            pa7.pt_start.y = center.y - axis1.y * (axis1_size+0.03) - normal.y*0.02;
            pa7.pt_start.z = center.z - axis1.z * (axis1_size+0.03) - normal.z*0.02;
            pa7.pt_end.x = pa7.pt_start.x + axis1.x/norm1 * action_distance;
            pa7.pt_end.y = pa7.pt_start.y + axis1.y/norm1 * action_distance;
            pa7.pt_end.z = pa7.pt_start.z;            
            actions.push_back(pa7);

            PushAction pa8;
            pa8.pt_start.x = center.x + axis2.x * (axis2_size+0.03) - normal.x*0.02;
            pa8.pt_start.y = center.y + axis2.y * (axis2_size+0.03) - normal.y*0.02;
            pa8.pt_start.z = center.z + axis2.z * (axis2_size+0.03) - normal.z*0.02;
            pa8.pt_end.x = pa8.pt_start.x - axis2.x/norm2 * action_distance;
            pa8.pt_end.y = pa8.pt_start.y - axis2.y/norm2 * action_distance;
            pa8.pt_end.z = pa8.pt_start.z;            
            actions.push_back(pa8);

            PushAction pa9;
            pa9.pt_start.x = center.x - axis2.x * (axis2_size+0.03) - normal.x*0.02;
            pa9.pt_start.y = center.y - axis2.y * (axis2_size+0.03) - normal.y*0.02;
            pa9.pt_start.z = center.z - axis2.z * (axis2_size+0.03) - normal.z*0.02;
            pa9.pt_end.x = pa9.pt_start.x + axis2.x/norm2 * action_distance;
            pa9.pt_end.y = pa9.pt_start.y + axis2.y/norm2 * action_distance;
            pa9.pt_end.z = pa9.pt_start.z;            
            actions.push_back(pa9);            

        }
    }

    KdTreeFLANN<PointXYZRGB> kdtree;
    kdtree.setInputCloud(cloud_all);
    actions_out.clear();
    for( size_t a=0; a<actions.size(); a++ )
    {
        if(IsValidAction(actions[a],kdtree)) actions_out.push_back(actions[a]);
    }

#if 0
    pcl::visualization::PCLVisualizer viewer("Segmentation Result");
    viewer.setSize(600,480);
    viewer.setPosition(600,0);
    viewer.setCameraPosition(0,0,0,1,0,-0.28,0,0,1);
    for( size_t a=0; a<actions.size(); a++ )
    {
        PointXYZ pt1;
        pt1.x = actions[a].pt_start.x;
        pt1.y = actions[a].pt_start.y;
        pt1.z = actions[a].pt_start.z;

        PointXYZ pt2;
        pt2.x = actions[a].pt_end.x;
        pt2.y = actions[a].pt_end.y;
        pt2.z = actions[a].pt_end.z;

        stringstream ss;
        ss << a;
        if(IsValidAction(actions[a],kdtree))
        {
            viewer.addLine(pt1, pt2, 0,1,0, ss.str() );
        }        
        else
        {
            viewer.addLine(pt1, pt2, 0,0,1, ss.str() );
        }
    }
    viewer.spin();
#endif
}

PushAction action;
PointXYZ pt_init(0.5,0.0,0.7);
void DrawAction(visualization::PCLVisualizer &viewer, PushAction &action, bool gray)
{
    static int idx=0;
    stringstream ss;
    ss << "line_" << idx++;
    PointXYZ pt1( action.pt_start.x,
                  action.pt_start.y,
                  action.pt_start.z  );
    PointXYZ pt2( action.pt_end.x,
                  action.pt_end.y,
                  action.pt_end.z  );
    
    PointXYZ pt_init;
    pt_init.x = action.pt_start.x;
    pt_init.y = action.pt_start.y;
    pt_init.z = action.pt_start.z + effector_length + 0.1;

    if( gray )
    {
        ss << "_init-start";
        viewer.addLine(pt_init, pt1, 0.2,0.2,0.2, ss.str() );
        ss << "_-end";
        viewer.addLine(pt1,     pt2, 0.2,0.2,0.2, ss.str() );
        viewer.setShapeRenderingProperties(
            visualization::PCL_VISUALIZER_LINE_WIDTH, 2, ss.str() );
    }
    else
    {
        ss << "_init-start";
        viewer.addLine(pt_init, pt1, 0,0,1, ss.str() );
        ss << "_-end";
        viewer.addLine(pt1,     pt2, 0,1,0, ss.str() );
        viewer.setShapeRenderingProperties(
            visualization::PCL_VISUALIZER_LINE_WIDTH, 4, ss.str() );
    }
}

static bool go_to_next = false;
static bool select_another_action = true;
static bool refresh_seg = false;
static bool close_fingers = false;
static bool open_fingers = false;
static bool running = true;
void Callback(const visualization::KeyboardEvent &event,
                           void* viewer_void)
{
    bool updated_action = false;
    visualization::PCLVisualizer *viewer
     = static_cast<visualization::PCLVisualizer *> (viewer_void);
    if (event.keyDown())
    {
        bool updated_action = false;
        if( event.getKeySym()=="Escape" )
        {
            running = false;            
        }
        else if( event.getKeySym()=="Return" )
        {
            go_to_next = true;
        }
        else if( event.getKeySym()=="space" )
        {            
            select_another_action = true;
        }
        else if( event.getKeySym()=="F5" )
        {            
            refresh_seg = true;
        }
        else if( event.getKeyCode()=='+'  )
        {
            open_fingers = true;
        }
        else if( event.getKeyCode()=='-'  )
        {
            close_fingers = true;
        }
        else if( event.getKeyCode()=='w' )
        {
            action.pt_start.x += 0.01;
            action.pt_end.x   += 0.01;
            pt_init.x = action.pt_start.x;
            updated_action = true;
        }
        else if( event.getKeyCode()=='s' )
        {
            action.pt_start.x -= 0.01;
            action.pt_end.x   -= 0.01;
            pt_init.x = action.pt_start.x;
            updated_action = true;
        }
        else if( event.getKeyCode()=='a' )
        {
            action.pt_start.y += 0.01;
            action.pt_end.y   += 0.01;
            pt_init.y = action.pt_start.y;
            updated_action = true;
        }
        else if( event.getKeyCode()=='d' )
        {
            action.pt_start.y -= 0.01;
            action.pt_end.y   -= 0.01;
            pt_init.y = action.pt_start.y;
            updated_action = true;
        }
        else if( event.getKeyCode()=='e' )
        {
            action.pt_start.z += 0.01;
            action.pt_end.z   += 0.01;
            pt_init.z = action.pt_start.z + effector_length + 0.1;
            updated_action = true;
        }
        else if( event.getKeyCode()=='c' )
        {
            action.pt_start.z -= 0.01;
            action.pt_end.z   -= 0.01;
            pt_init.z = action.pt_start.z + effector_length + 0.1;
            updated_action = true;
        }
        else if( event.getKeyCode()=='W' )
        {
            action.pt_end.x   += 0.01;
            updated_action = true;
        }
        else if( event.getKeyCode()=='S' )
        {
            action.pt_end.x   -= 0.01;
            updated_action = true;
        }
        else if( event.getKeyCode()=='A' )
        {
            action.pt_end.y   += 0.01;
            updated_action = true;
        }
        else if( event.getKeyCode()=='D' )
        {
            action.pt_end.y   -= 0.01;
            updated_action = true;
        }
        else if( event.getKeyCode()=='E' )
        {
            action.pt_end.z   += 0.01;
            updated_action = true;
        }
        else if( event.getKeyCode()=='C' )
        {
            action.pt_end.z   -= 0.01;
            updated_action = true;
        }


        if( updated_action )
        {         
            viewer->removeAllShapes();
            DrawAction(*viewer,action,false);
        }
    }
}

void Callback_image(const sensor_msgs::Image::ConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr_image;
    try
    {
        cv_ptr_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::imshow("image", cv_ptr_image->image);    
    cv::waitKey(300); 
}

int main( int argc, char** argv )
{
    ros::init(argc,argv,"random_push");
    ros::NodeHandle nh;
    ParseParam(nh);
    PrintParam();

    ros::AsyncSpinner spinner(1);
    spinner.start();

    ros::ServiceClient clt_lccp_2Dseg
     = nh.serviceClient<rl_msgs::seg_scene_srv>("/segmentation/lccp_2Dseg");

    CameraRecorder rec(nh, outdir, "scene01");

    Pusher pusher;
    //pusher.MoveHome();
    pusher.MoveInit();
    cout << "Press [Enter] when it is good to close fingers" << endl;    
    fflush(stdin);
    getchar();
    pusher.CloseFingers();

    ROS_INFO_STREAM("READY");

    ros::Subscriber sub_image;
    sub_image = nh.subscribe("/rgb/image",10,Callback_image);
    cv::namedWindow("image");
    cv::moveWindow("image", 600,0);

    pcl::visualization::PCLVisualizer viewer("Segmentation Result");
    viewer.setSize(600,480);
    viewer.setPosition(0,0);
    viewer.setCameraPosition(0,0,0,1,0,-0.28,0,0,1);
    viewer.registerKeyboardCallback(Callback, (void*)&viewer);

    int idx=0;
    string param;    
    while(running)
    {   
        PointCloud<PointXYZRGB>::Ptr cloud_input(new PointCloud<PointXYZRGB>);        
        
        ROS_INFO_STREAM("Segmentation - Request");
        rl_msgs::seg_scene_srv srv;
        srv.request.use_camera = true;
        srv.request.camera_RT = camera_RT;
        if( clt_lccp_2Dseg.call(srv) )
        {
            fromRLMsg(srv.response.segscene, cloud_input);
        }
        else
        {
            ROS_ERROR_STREAM("Cannot segment using camera");
            break;
        }
        ROS_INFO_STREAM("Segmentation - Done");
        
        vector<PushAction> actions;
        GeneratePushActions(srv.response.segscene, actions);
        if( actions.size()==0 )
        {
            ROS_WARN_STREAM("No valid actions");
            cout << "Press [Enter]" << endl;    
            fflush(stdin);
            getchar();
            continue;
        }

        viewer.removeAllPointClouds();
        viewer.removeAllShapes();

        PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);
        transformPointCloud(*cloud_input, *cloud, tf_cam2world);
        viewer.addPointCloud(cloud, "cloud");
       
        int a=0;
        select_another_action = true;
        while( !go_to_next && running )
        {
            viewer.spinOnce(3);
            if( select_another_action && running )
            {   
                a = (int)rand() % (int)actions.size();
                action.pt_start.x = actions[a].pt_start.x;
                action.pt_start.y = actions[a].pt_start.y;
                action.pt_start.z = actions[a].pt_start.z;
                action.pt_end.x   = actions[a].pt_end.x;
                action.pt_end.y   = actions[a].pt_end.y;
                action.pt_end.z   = actions[a].pt_end.z;

                pt_init.x = action.pt_start.x;
                pt_init.y = action.pt_start.y;
                pt_init.z = action.pt_start.z + effector_length + 0.1;
                                
                viewer.removeAllShapes();
                for( size_t i=0; i<actions.size(); i++ )
                {
                    if( i!=a ) DrawAction(viewer, actions[i], true);    
                }
                DrawAction(viewer, action, false);

                select_another_action = false;
            }
            if( refresh_seg )
            {
                PointCloud<PointXYZRGB>::Ptr cloud_input_new(new PointCloud<PointXYZRGB>);

                rl_msgs::seg_scene_srv srv_new;
                srv_new.request.use_camera = true;
                srv_new.request.camera_RT = camera_RT;
                if( clt_lccp_2Dseg.call(srv_new) )
                {
                    fromRLMsg(srv_new.response.segscene, cloud_input_new);
                }
                else
                {
                    running = false;
                    ROS_ERROR_STREAM("Cannot segment using camera");                    
                }
                refresh_seg = false;

                GeneratePushActions(srv_new.response.segscene, actions);

                viewer.removeAllPointClouds();
                viewer.removeAllShapes();

                PointCloud<PointXYZRGB>::Ptr cloud_new(new PointCloud<PointXYZRGB>);
                transformPointCloud(*cloud_input_new, *cloud_new, tf_cam2world);
                viewer.addPointCloud(cloud_new, "cloud");

                select_another_action = true;
            }
            if( open_fingers )
            {
                pusher.OpenFingers();
                open_fingers = false;
            }
            if( close_fingers )
            {
                pusher.CloseFingers();
                close_fingers = false;
            }
        }
        if( running==false )
        {
            pusher.MoveHome();
            break;
        }

        go_to_next = false;

        // Move Arm

        /*
        ROS_INFO_STREAM("[STEER] init");
        if( !pusher.MoveInit() )
        { 
            ROS_WARN_STREAM("Failed Move to the init Position: " 
                               << "x: " << pt_init.x
                               << "y: " << pt_init.y
                               << "z: " << pt_init.z               );
            continue;
        }
        
        ROS_INFO_STREAM("[STEER] pre-start"
                         << " x:" << action.pt_start.x
                         << " y:" << action.pt_start.y
                         << " z:" << action.pt_start.z + effector_length*2);
        if( !pusher.Steer( action.pt_start.x,
                           action.pt_start.y,
                           action.pt_start.z + effector_length*2) )
        {
            ROS_WARN_STREAM("Failed Move to the start+init Position: " 
                            << "x: " << action.pt_start.x
                            << "y: " << action.pt_start.y
                            << "z: " << action.pt_start.z + effector_length*2);
            continue;
        }
        */
        if( !pusher.PrePush( 
            action.pt_start.x, 
            action.pt_start.y, 
            action.pt_start.z + effector_length,
            action.pt_end.x, 
            action.pt_end.y, 
            action.pt_end.z + effector_length) )
        {
            ROS_WARN_STREAM("Failed Move to the pre-start Position: " 
                            << "x: " << action.pt_start.x
                            << "y: " << action.pt_start.y
                            << "z: " << action.pt_start.z + effector_length*2);
            continue;
        }

        cout << "Press [Enter]" << endl;
        fflush(stdin);
        getchar();

        // Take snapshots
        stringstream ss;
        ss << "scene" << idx++;
        rec.SetPrefix(ss.str());
        rec.ResetCounter();
        rec.StartSave(0.1);

        /*
        ROS_INFO_STREAM("[STEER] start"
                        << " x:" << action.pt_start.x
                        << " y:" << action.pt_start.y
                        << " z:" << action.pt_start.z + effector_length );
        if( !pusher.Steer( action.pt_start.x, 
                           action.pt_start.y, 
                           action.pt_start.z + effector_length) )
        {
            ROS_WARN_STREAM("Failed Move to the start Position: " 
                               << " x:" << action.pt_start.x
                               << " y:" << action.pt_start.y
                               << " z:" << action.pt_start.z + effector_length    );
            rec.StopSave();
            pusher.MoveInit();
            continue;
        }

        ROS_INFO_STREAM("[STEER] end" 
                        << " x:" << action.pt_end.x
                        << " y:" << action.pt_end.y
                        << " z:" << action.pt_end.z + effector_length );
        if( !pusher.Steer( action.pt_end.x,   
                           action.pt_end.y,   
                           action.pt_end.z   + effector_length) )
        {
            ROS_WARN_STREAM("Failed Move to the end Position: " 
                               << " x:" << action.pt_end.x
                               << " y:" << action.pt_end.y
                               << " z:" << action.pt_end.z      );
            rec.StopSave();
            pusher.MoveInit();
            continue;
        }
        */
        if( !pusher.Push( 
            action.pt_start.x, 
            action.pt_start.y, 
            action.pt_start.z + effector_length,
            action.pt_end.x, 
            action.pt_end.y, 
            action.pt_end.z + effector_length) )
        {
            ROS_WARN_STREAM("Failed Move to the Position: " 
                               << " x:" << action.pt_end.x
                               << " y:" << action.pt_end.y
                               << " z:" << action.pt_end.z 
                               << " => "
                               << " x:" << action.pt_end.x
                               << " y:" << action.pt_end.y
                               << " z:" << action.pt_end.z      );
            rec.StopSave();
            pusher.MoveInit();
            continue;
        }
        
        rec.StopSave();        
    
        // save the action
        ofstream f_action;
        f_action.open (outdir + "/" + ss.str() + ".action.txt");
        f_action << "<start "
                 << "x=" << action.pt_start.x << " "
                 << "y=" << action.pt_start.y << " "
                 << "z=" << action.pt_start.z << " />" << endl;
        f_action << "<end "
                 << "x=" << action.pt_end.x << " "
                 << "y=" << action.pt_end.y << " "
                 << "z=" << action.pt_end.z << " />" << endl;
        f_action.close();
    }

    return 0;
}
